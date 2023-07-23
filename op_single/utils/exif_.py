from PIL import Image, ExifTags

tags = ExifTags.TAGS

def get_focal(exif):
    for k in exif_data.keys():
        if 'FocalLength'==tags[k]:
            print(exif_data[k], '\t', tags[k])

def get_exif(exif):
    for k in exif_data.keys():
        if tags[k] != 'MakerNote':
            print(tags[k], '\t', exif_data[k])
        else:
            # decoded_string = exif_data[k].decode('UNICODE', errors='ignore')
            # print(decoded_string)
            pass

exif_data = Image.open(r'.\img\batch_warp\origin\DSC00149.JPG')._getexif()
get_exif(exif_data)

# exif_data = Image.open('.\img\\2.jpg')._getexif()
# get_focal(exif_data)
# exif_data = Image.open('.\img\\3.jpg')._getexif()
# get_focal(exif_data)
# exif_data = Image.open('.\img\\4.jpg')._getexif()
# get_focal(exif_data)
# exif_data = Image.open('.\img\\5.jpg')._getexif()
# get_focal(exif_data)

# print(ExifTags.TAGS)
