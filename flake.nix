{
  description = "Gufo with the VoxCPM extension";
  inputs.gufo.url = "github:gufo-org/gufo";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default = gufo.packages.x86_64-linux.default.override {
      extensions = [ self.outPath ];
    };
  };
}
