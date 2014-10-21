#-*-encoding:gbk-*-
class feature_extract:
    file_loader = file_basic_loader()
    feature_extractor = feature_basic_extractor()
    feature_dumper = feature_basic_dumper()
    def init(self, in_file_loader, in_feature_extractor, in_feature_dumper):
        self.file_loader = in_file_loader
        self.feature_extractor = in_feature_extractor
        self.feature_dumper = in_feature_dumper
    
    def process(self, dumper_file):
        ret = self.file_loader(stock_file)
        if ret != 0:
            sys.stderr.write("file loader error")
            sys.exit(1)

        ret = self.feature_extractor()
        if ret != 0:
            sys.stderr.write("feature extract error")
            sys.exit(1)

        ret = self.feature_dumper(dumper_file)
        if ret != 0:
            sys.stderr.write("feature dump error")
            sys.exit(1)
        return 0


if __name__ == "__main__":
    stock_file = sys.argv[1]
    dump_file = sys.argv[2]
    exactor = feature_extract()
    exactor.process(stock_file, dump_file)

    
