import os, zlib, struct, urllib.request
from PIL import Image
from io import BytesIO

def decode_png_idat(idat_data, width, height, header=[8, 2, 0, 0, 0]):
    def write_chunk(chunk_type, chunk_data):
        f.write(struct.pack(">I", len(chunk_data)))
        f.write(chunk_type)
        f.write(chunk_data)
        f.write(struct.pack(">I", zlib.crc32(chunk_type + chunk_data)))

    f = BytesIO()
    f.write(b"\x89PNG\r\n\x1a\n")
    write_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, *header))
    write_chunk(b"IDAT", idat_data)
    write_chunk(b"IEND", b"")
    f.seek(0)
    return Image.open(f)

def download_test_images():
    # backup URL in case the other URL is down
    # https://web.archive.org/web/20211023132308/https://sjtrny.com/files/10.1109_DICTA.2012.6411686.pdf
    url = "https://sjtrny.com/files/10.1109_DICTA.2012.6411686.pdf"
    filename = url.split("/")[-1]

    if not os.path.exists(filename):
        print(f"Downloading...")
        urllib.request.urlretrieve(url, filename)

    with open(filename, "rb") as f:
        data = f.read()

    # Three example images with hardcoded offsets into the PDF
    example_image_data = [
        ("trimap.png", data[2828574:2828574+13271]),
        ("ground_truth_alpha.png", data[5275869:5275869+211014]),
        ("image.png", data[6939898:6939898+840881]),
    ]
    width = 800
    height = 563

    for filename, image_data in example_image_data:
        image = decode_png_idat(image_data, width, height)
        image.save(filename)

if __name__ == "__main__":
    download_test_images()
