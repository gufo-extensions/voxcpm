{
  description = "Gufo with the VoxCPM extension";
  inputs.gufo.url = "github:gufo-org/gufo/2845939b2a7d61243d407c8816c6d42652512e50";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self.outPath ];
    };
  };
}
