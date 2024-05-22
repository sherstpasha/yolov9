from detect_function import detect_image

weights_path = r"C:\Users\user\Desktop\ddata\exp3\weights\best.pt"
image_path = r"C:\Users\user\Desktop\test_images"
results = detect_image(weights=weights_path, source=image_path)

for res in results:
    print(res)