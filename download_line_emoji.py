import os
import requests
from argparse import ArgumentParser

if not os.path.exists("output"):
    os.makedirs("output")


class EmojiDownload:
    PNG_PATTERN = "https://stickershop.line-scdn.net/stickershop/v1/sticker/"
    # background-image:url(https://stickershop.line-scdn.net/sticonshop/v1/sticon/65f8e87665bd7b66653cb82a/iPhone/005.png?v=1);

    def __init__(self, url):
        self.url = url
        self.html = None
        self.name = "placeholder"
    
    def read_html(self, cache=False):
        if not os.path.isfile("output/line_emoji.html") or not cache:
            r = requests.get(self.url)
            with open("output/line_emoji.html", "w") as f:
                f.write(r.text)
            
            self.html = r.text

        else:
            with open("output/line_emoji.html", "r") as f:
                self.html = f.read()
        
        title_find = self.html.find("sticker-name-title")

        if title_find >= 0:
            title_find = self.html.find(">", title_find) + 1
            title_end = self.html.find("</", title_find)

            self.name = self.html[title_find:title_end]

        else:
            title_find = self.html.find("\"name\": \"")
            title_find = self.html.find("\"", title_find) + 9

            title_end = self.html.find("\"", title_find)

            self.name = self.html[title_find:title_end]
    
    def download(self):
        os.makedirs(f"output/{self.name}", exist_ok=True)

        find_list = set()

        find_index = self.html.find(self.PNG_PATTERN)

        while find_index != -1:
            end = self.html.find(".png", find_index)
            find_list.add(self.html[find_index:end+4])
            find_index = self.html.find(self.PNG_PATTERN, end)

        for i, image_url in enumerate(find_list):
            img_data = requests.get(image_url).content

            with open(f'output/{self.name}/image-{i}.png', 'wb') as handler:
                handler.write(img_data)


if __name__ == "__main__":
    # smaple_url = "https://store.line.me/stickershop/product/28718466/en"
    parser = ArgumentParser()
    parser.add_argument("url", help="The URL of the LINE emoji page.")

    args = parser.parse_args()

    emoji = EmojiDownload(args.url)
    emoji.read_html()
    print(f"Downloading \"{emoji.name}\" ...")
    emoji.download()
