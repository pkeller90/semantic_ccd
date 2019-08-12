from pathlib import Path
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import javalang

# This might have to be changed for OSes different than ubuntu
FONT_PATH = '/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf'

def load_code(path):
    code = []
    with open(path, "r") as fin:
        for l in fin:
            code.append(l)
    return "".join(code)


def update_pos(prev_pos, prev_token_pos, new_token_pos, prev_token_size, line_height):
    X_SCALE = 1

    pos_delta = (new_token_pos[0] - prev_token_pos[0],
                 new_token_pos[1] - prev_token_pos[1])

    if pos_delta[1] == 0:
        nx = prev_pos[0] + prev_token_size[0] + X_SCALE * (pos_delta[0]-1)
        ny = prev_pos[1]
    else:
        nx = X_SCALE * new_token_pos[0]
        ny = prev_pos[1] + line_height + 3

    #print(nx)
    #print(ny)
    #print("==========")

    return (nx, ny), max(line_height, prev_token_size[1])


def keywords_picto(code):
    code = code.replace("\t", "    ")
    background = (255,255,255)

    fontsize = 14
    font = ImageFont.truetype(FONT_PATH, fontsize)

    # load keyword images
    curr_path = "keywords_picto"
    keyword_forms_dir = "%s/keyword_forms" % curr_path
    keyword_forms = {}
    for keyword in os.listdir(keyword_forms_dir):
        if keyword.endswith(".png"):
            img = Image.open(keyword_forms_dir + "/" + keyword)
            ratio = float(fontsize) / img.size[1]
            keyword_forms[keyword.replace(".png", "")] = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.ANTIALIAS)

    width, height = ImageDraw.Draw(Image.new('RGBA', (1,1), background)).textsize(code, font)
    image = Image.new('RGBA', (int(width * 1.1), int(height * 1.1)), background)
    draw = ImageDraw.Draw(image)


    lines = code.replace("\t", "    ")
    tokens = javalang.tokenizer.tokenize(lines)

    prev_token_pos = (0,0)
    prev_token_size = (0,0)
    prev_pos = (0,0)
    line_height = 0
    for w in tokens:
        #print(vars(w))
        #continue
        new_token_pos = (w.position[1]-1, w.position[0]-1)
        npos, line_height = update_pos(prev_pos, prev_token_pos, new_token_pos, prev_token_size, line_height)
        #print(wpos)
        w = w.value
        if w in keyword_forms:
            #print(w)
            ki = keyword_forms[w]

            image.paste(ki, (npos[0], npos[1], npos[0]+ki.size[0], npos[1]+ki.size[1]))
            prev_token_size = ki.size
        else:
            textsize = draw.textsize(w, font)
            draw.text((npos[0], npos[1]), w+" ", fill='black', font=font)
            prev_token_size = textsize
        prev_token_pos = new_token_pos
        prev_pos = npos
    return image

def from_to_file(in_path, out_path):
    code = load_code(in_path)
    image = keywords_picto(code)
    image.save(out_path, )


if __name__ == "__main__":
    from_to_file("<path>/visualizations/example.java", "<path>/visualizations/example_icon_high.png") # Path must be set
