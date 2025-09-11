def rgb_to_hex(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

print(rgb_to_hex(59, 95, 33))  # 输出 #003B21