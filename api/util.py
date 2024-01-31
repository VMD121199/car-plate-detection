import string
import easyocr
import csv
import time
import datetime

# Initialize the OCR reader
reader = easyocr.Reader(["en"], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5"}

dict_int_to_char = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S"}

region_mapping = {}

def load_region_mapping_from_csv(csv_path):
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            prefixes = [prefix.strip() for prefix in row[0].split(',')]
            region = row[1].strip()
            for prefix in prefixes:
                region_mapping[prefix] = region


def get_license_plate_region(license_plate_text):
    return region_mapping.get(license_plate_text[:2], "Unknown")


"""

#for videos

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()

"""

# for images


def write_csv(results, output_path):
    with open(output_path, "w", newline="") as csvfile:
        fieldnames = [
            "image_idx",
            "license_plate_bbox",
            "bbox_score",
            "license_number",
            "text_score",
            "detection_time"
            "region",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for image_idx in results.keys():
            for class_id in results[image_idx].keys():
                if (
                    "license_plate" in results[image_idx][class_id].keys()
                    and "text"
                    in results[image_idx][class_id]["license_plate"].keys()
                ):

                    license_plate_info = results[image_idx][class_id][
                        "license_plate"
                    ]

                    writer.writerow({
                        'image_idx': datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'),
                        'license_plate_bbox': '[{} {} {} {}]'.format(
                            license_plate_info['bbox'][0],
                            license_plate_info['bbox'][1],
                            license_plate_info['bbox'][2],
                            license_plate_info['bbox'][3]),
                        'bbox_score': license_plate_info['bbox_score'],
                        'license_number': license_plate_info['text'],
                        'text_score': license_plate_info['text_score'],
                        'detection_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'region': license_plate_info['region']
                    })


#### Specific to car plates in UK

def license_complies_format(text):
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_



def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    license_plate_text = ""
    score = 0

    for detection in detections:
        bbox, text, detection_score = detection
        text = text.upper().replace(' ', '')

        # Accumulate characters to form the complete license plate text
        license_plate_text += text
        score = max(score, detection_score)

    if license_complies_format(license_plate_text):
        formatted_license_plate = format_license(license_plate_text)
        return formatted_license_plate, score
    else:
        return license_plate_text, score


def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
