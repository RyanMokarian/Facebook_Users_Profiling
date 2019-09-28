from src.util import Utils

util = Utils()


def run_classifiers():
    raw_data = util.read_csv('./data/Train/Profile/Profile.csv')[1:]
    data = []
    for row in raw_data:
        try:
            gender = float(row[3])
            age = float(row[2])
            if age >= 0 and (gender == 1 or gender == 0):
                data.append(row)
        except ValueError:
            print("Bad data, skipping record")
            continue

    females = 0
    age_groups = {"xx-24": 0, "25-34": 0, "35-49": 0, "50-xx": 0}
    for e in data:
        gender = float(e[3])
        if gender > 0:
            females += 1
        age = float(e[2])
        if age < 25:
            age_groups["xx-24"] += 1
        elif age < 35:
            age_groups["25-34"] += 1
        elif age < 50:
            age_groups["35-49"] += 1
        else:
            age_groups["50-xx"] += 1

    most_common_age = max(age_groups, key=age_groups.get)

    males = len(data) - females
    most_common_gender = "female" if females > males else "male"

    return most_common_age, most_common_gender


def main():
    # TODO Training data goes here
    classification_data = run_classifiers()
    model = util.read_json("./resources/model.json")
    model["age_group"] = classification_data[0]
    model["gender"] = classification_data[1]
    util.write_json_to_directory(model, "model.json")
    ss = ""
    pass


if __name__ == "__main__":
    main()
