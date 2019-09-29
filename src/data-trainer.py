from util import Utils

util = Utils()


def run_classifiers():
    """
    This method runs the classifier to find an average for age and gender
    """
    raw_data = util.read_csv('../data/Train/Profile/Profile.csv')[1:]
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


def calculate_personality_traits():
    """
    This method extract data from the CSV file to find their average
    """
    raw_data = util.read_csv('../data/Train/Profile/Profile.csv')[1:]
    extrovert = 0
    neurotic = 0
    agreeable = 0
    conscientious = 0
    open = 0
    for row in raw_data:
        try:
            extrovert += float(row[6])
            neurotic += float(row[8])
            agreeable += float(row[7])
            conscientious += float(row[5])
            open += float(row[4])
        except ValueError:
            print("Bad data, skipping record")
            continue
    extrovert = find_average(extrovert, len(raw_data))
    neurotic = find_average(neurotic, len(raw_data))
    agreeable = find_average(agreeable, len(raw_data))
    conscientious = find_average(conscientious, len(raw_data))
    open = find_average(open, len(raw_data))
    return extrovert, neurotic, agreeable, conscientious, open


def find_average(total, amount):
    """
    This method find the average of each of the personality traits
    """
    return total / amount


def main():
    classification_data = run_classifiers()
    extrovert, neurotic, agreeable, conscientious, open = calculate_personality_traits()
    model = util.read_json("./resources/model.json")
    model["age_group"] = classification_data[0]
    model["gender"] = classification_data[1]
    model["extrovert"] = round(extrovert, 2)
    model["neurotic"] = round(neurotic, 2)
    model["agreeable"] = round(agreeable, 2)
    model["conscientious"] = round(conscientious, 2)
    model["open"] = round(open, 2)
    util.write_json_to_directory(model, "model.json")
    pass


if __name__ == "__main__":
    main()
