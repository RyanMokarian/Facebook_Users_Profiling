from src.util import Utils


def main():
    # TODO Training data goes here
    util = Utils()
    aaa = util.read_json("model.json")
    aaa["age_group"] = 22
    util.write_json_to_directory(aaa, "model.json")
    ss = ""
    pass


if __name__ == "__main__":
    main()
