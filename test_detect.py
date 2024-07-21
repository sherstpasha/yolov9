from detect_function_dual import detect_image_dual

weight_path = r"C:\Users\pasha\OneDrive\Рабочий стол\yolo_weights\yolo_rusal.pt"
source = r"C:\Users\pasha\OneDrive\Рабочий стол\photo_2024-06-18_14-25-05.jpg"


res = detect_image_dual(weight_path, source, device="cpu")

print(res)
