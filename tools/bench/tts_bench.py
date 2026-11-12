#!/usr/bin/env python3
"""VoxCPM2 single-request benchmark: Gufo `serve tts` vs audio.cpp server.

Usage: tts_bench.py gufo|audiocpp OUTPUT_JSON

Environment: VOXCPM_MODEL (the Hugging Face directory), GUFO (the gufo
binary), AUDIOCPP_SERVER and AUDIOCPP_MODEL (a VoxCPM2 GGUF for audio.cpp),
REFERENCE (a WAV with a .txt transcript beside it, without the extension),
PORT.

Launches the server with a cold file cache, times readiness, then per
variant (a short sentence, and the 37-word paragraph of Gufo's TTS benchmarks
with an automatic voice, voice design and a clone with its transcript) one
warmup and three timed WAV requests (median) plus one SSE request for first
audio, seed 42. Voice design puts the description in parentheses before the
text on both servers, which is VoxCPM2's own syntax. Records the server's
resident memory and the GPU memory it added.
"""
import base64, json, os, signal, statistics, struct, subprocess, sys, time, urllib.request

PARAGRAPH = ("When the first light reaches the harbor, speak gently, with quiet confidence "
             "and a warm, resonant tone. The boats are still resting beside the old wooden "
             "pier, and the town is slowly waking to another bright morning.")
FIXTURE = os.environ.get("REFERENCE", "tests/functional/fixtures/audio/clear-english-voice")
MODEL_DIR = os.environ.get("VOXCPM_MODEL", "")
GUFO = os.environ.get("GUFO", "gufo")
AUDIOCPP_SERVER = os.environ.get("AUDIOCPP_SERVER", "audiocpp_server")
AUDIOCPP_MODEL = os.environ.get("AUDIOCPP_MODEL", "")
PORT = int(os.environ.get("PORT", "18201"))
INSTRUCT = "A young woman with a calm, low voice and a British accent"
SHORT = "The little brown dog ran across the garden."
GPU = "/sys/class/drm/card0/device"


def gpu_memory():
    """Bytes of VRAM and GTT in use."""
    total = 0
    for name in ("mem_info_vram_used", "mem_info_gtt_used"):
        with open(os.path.join(GPU, name)) as stream:
            total += int(stream.read())
    return total


def resident(pid):
    with open(f"/proc/{pid}/status") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def drop_cache(paths):
    for root in paths:
        files = [root] if os.path.isfile(root) else [
            os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
        for path in files:
            with open(path, "rb") as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def post(path, body, stream=False):
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        if not stream:
            data = response.read()
            return time.perf_counter() - start, None, data
        first = None
        chunks = []
        for line in response:
            if line.startswith(b"data:"):
                try:
                    event = json.loads(line[5:])
                except ValueError:
                    continue  # such as "[DONE]"
                audio = event.get("audio") or event.get("delta")
                if event.get("type") == "speech.audio.delta" and audio:
                    if first is None:
                        first = time.perf_counter() - start
                    chunks.append(base64.b64decode(audio))
        return time.perf_counter() - start, first, b"".join(chunks)


def wav_data(data):
    """The PCM bytes of a WAV file's data chunk."""
    at = 12
    while at + 8 <= len(data):
        chunk, size = data[at:at + 4], struct.unpack("<I", data[at + 4:at + 8])[0]
        if chunk == b"data":
            return data[at + 8:at + 8 + size]
        at += 8 + size + (size & 1)
    raise ValueError("no data chunk")


def wav_seconds(data):
    # Walk RIFF chunks for fmt and data.
    rate = channels = bits = None
    at = 12
    while at + 8 <= len(data):
        chunk, size = data[at:at + 4], struct.unpack("<I", data[at + 4:at + 8])[0]
        if chunk == b"fmt ":
            channels, rate = struct.unpack("<HI", data[at + 10:at + 16])
            bits = struct.unpack("<H", data[at + 22:at + 24])[0]
        elif chunk == b"data":
            size = min(size, len(data) - at - 8)
            return size / (rate * channels * bits // 8)
        at += 8 + size + (size & 1)
    raise ValueError("no data chunk")


def requests_for(server):
    clone_text = open(FIXTURE + ".txt").read().strip()
    designed = "(" + INSTRUCT + ")" + PARAGRAPH
    if server == "gufo":
        base = {"model": "voxcpm2", "input": PARAGRAPH, "seed": 42}
        return {"short": {**base, "input": SHORT, "voice": "auto"},
                "auto": {**base, "voice": "auto"},
                "design": {**base, "voice": "auto", "input": designed},
                "clone": {**base, "voice": "narrator"}}, "voxcpm2"
    base = {"model": "voxcpm2", "input": PARAGRAPH, "seed": 42}
    return {"short": {**base, "input": SHORT}, "auto": base,
            "design": {**base, "input": designed},
            "clone": {**base, "voice": "narrator", "reference_text": clone_text}}, "voxcpm2-stream"


def launch(server, log, output_path):
    if server == "gufo":
        command = [GUFO, "serve", "tts",
                   "--model", MODEL_DIR, "--host", "127.0.0.1", "--port", str(PORT),
                   "--voice", f"narrator={FIXTURE}.wav", "--voice-text", f"narrator={FIXTURE}.txt"]
        ready = "/ready"
        drop_cache([MODEL_DIR])
    else:
        scratch = os.path.dirname(os.path.abspath(output_path))
        voices = os.path.join(scratch, "voices")
        os.makedirs(voices, exist_ok=True)
        if not os.path.exists(os.path.join(voices, "narrator.wav")):
            os.symlink(os.path.abspath(FIXTURE + ".wav"), os.path.join(voices, "narrator.wav"))
        config = {"host": "127.0.0.1", "port": PORT, "backend": "vulkan", "device": 0, "threads": 4,
                  "lazy_load": False, "voice_dir": voices,
                  "models": [{"id": "voxcpm2", "family": "voxcpm2", "path": AUDIOCPP_MODEL,
                              "task": "tts", "mode": "offline"},
                             {"id": "voxcpm2-stream", "family": "voxcpm2", "path": AUDIOCPP_MODEL,
                              "task": "tts", "mode": "streaming",
                              "default_request_options": {"retry_badcase": False}}]}
        path = os.path.join(scratch, "audiocpp_server.json")
        json.dump(config, open(path, "w"))
        command = [AUDIOCPP_SERVER, "--config", path]
        ready = "/health"
        drop_cache([AUDIOCPP_MODEL])
    start = time.perf_counter()
    process = subprocess.Popen(["choom", "-n", "1000", "--"] + command, stdout=log, stderr=subprocess.STDOUT)
    while True:
        if process.poll() is not None:
            raise RuntimeError("server exited")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{ready}", timeout=2) as response:
                if response.status == 200:
                    return process, time.perf_counter() - start
        except Exception:
            time.sleep(0.05)


def main():
    server, output = sys.argv[1], sys.argv[2]
    result = {"server": server, "paragraph": PARAGRAPH, "variants": {}}
    with open(output + ".server.log", "w") as log:
        gpu_before = gpu_memory()
        process, ready = launch(server, log, output)
        result["ready_seconds"] = ready
        try:
            requests, stream_model = requests_for(server)
            for name, body in requests.items():
                entry = {}
                first_time, _, first_wav = post("/v1/audio/speech", body)
                entry["first_request_seconds"] = first_time
                times, wavs = [], []
                for _ in range(3):
                    elapsed, _, data = post("/v1/audio/speech", body)
                    times.append(elapsed)
                    wavs.append(data)
                entry["wav_seconds"] = statistics.median(times)
                entry["wav_runs"] = times
                entry["audio_seconds"] = wav_seconds(wavs[0])
                entry["rtf"] = entry["wav_seconds"] / entry["audio_seconds"]
                entry["repeatable"] = all(w == wavs[0] for w in wavs) and first_wav == wavs[0]
                stream_body = {**body, "model": stream_model if server == "audiocpp" else body["model"],
                               "response_format": "pcm", "stream_format": "sse"}
                complete, first, pcm = post("/v1/audio/speech", stream_body, stream=True)
                entry["first_pcm_seconds"] = first
                entry["complete_pcm_seconds"] = complete
                entry["pcm_equals_wav"] = pcm == wav_data(wavs[0])
                result["variants"][name] = entry
                print(server, name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                                for k, v in entry.items() if k != "wav_runs"}), flush=True)
            result["resident_bytes"] = resident(process.pid)
            result["gpu_bytes"] = gpu_memory() - gpu_before
            print(server, "resident %.0f MiB, GPU %.0f MiB" % (
                result["resident_bytes"] / 2**20, result["gpu_bytes"] / 2**20), flush=True)
        finally:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
    json.dump(result, open(output, "w"), indent=1)


if __name__ == "__main__":
    main()
