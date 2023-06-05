import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import urllib.parse
import re


class Lazada:
    def __init__(self, chromedriver, headless=True) -> None:
        self.driver = self.setup(chromedriver, headless)
        self.data = []

    def setup(self, chromedriver, headless):
        opt = webdriver.ChromeOptions()
        opt.add_experimental_option('excludeSwitches', ['enable-logging'])
        if headless:
            opt.add_argument("--window-size=2560,1440")
            opt.add_argument('--ignore-certificate-errors')
            opt.add_argument('--allow-running-insecure-content')
            opt.add_argument("--disable-extensions")
            opt.add_argument("--proxy-server='direct://'")
            opt.add_argument("--proxy-bypass-list=*")
            opt.add_argument("--start-maximized")
            opt.add_argument('--headless')
            opt.add_argument('--disable-gpu')
            opt.add_argument('--disable-dev-shm-usage')
            opt.add_argument('--no-sandbox')
            user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'
            opt.add_argument(f'user-agent={user_agent}')

        return webdriver.Chrome(executable_path=chromedriver, options=opt)

    def get_details(self, detail_container, category, rank):
        # Scrape to get all parameters
        detail = dict()
        detail['rank'] = rank
        detail['category'] = category
        # Name
        try:
            name = detail_container.find_elements(By.XPATH, "./div")[1].find_element(
                By.XPATH, ".//a").get_attribute("title")
            detail['name'] = name
        except Exception as e:
            detail['name'] = None
        # Price
        try:
            price = detail_container.find_elements(
                By.XPATH, "./div")[2].find_element(
                By.XPATH, ".//span").get_attribute("innerHTML")
            price = float(re.sub('[^0-9]', '', price))
            detail['price'] = price
        except Exception as e:
            detail['price'] = None
        # Location
        try:
            location = detail_container.find_element(
                By.XPATH, ".//span[contains(text(),'Kota') or contains(text(),'Kab.')]").get_attribute("innerHTML")
            detail['location'] = location
        except Exception as e:
            detail['location'] = None
        # # Rating
        try:
            rating = detail_container.find_elements(
                By.XPATH, ".//i[@class = '_9-ogB Dy1nx']")
            rating = float(len(rating))
            detail['rating'] = rating
        except Exception as e:
            detail['rating'] = None

        # # Sold
        try:
            sold = detail_container.find_element(
                By.XPATH, ".//span[contains(text(),'Terjual')]").get_attribute("innerHTML")
            if ("rb" in sold):
                sold = int(re.sub('[^0-9]', '', sold))
                sold = sold * 1000
            else:
                sold = int(re.sub('[^0-9]', '', sold))
            detail['sold'] = sold
        except Exception as e:
            detail['sold'] = None

        return detail

    def search(self, cat):
        self.data = []
        cat = re.sub(r'[^\w\s]', '', cat)

        url_safe_cat = urllib.parse.quote(cat)
        url_safe_cat = url_safe_cat.replace("%20", "+")
        url = f"https://www.lazada.co.id/catalog/?q={url_safe_cat}"
        # print(f'Scraping for category {cat}..')
        self.driver.get(url)

        # for i in range(2):
        #     time.sleep(1)
        containers = WebDriverWait(self.driver, 100).until(EC.presence_of_all_elements_located(
            (By.XPATH, "//div[@data-qa-locator='product-item']")))

        for index, container in enumerate(containers):
            detail_container = container.find_element(By.TAG_NAME, "div").find_element(
                By.TAG_NAME, "div").find_elements(By.XPATH, "./div")[1]
            details = self.get_details(detail_container, cat, index)
            try:
                links = detail_container.find_elements(By.XPATH, "./div")[1].find_element(
                    By.XPATH, ".//a")
                url = links.get_attribute("href")
                details['url'] = url

                image = container.find_element(
                    By.XPATH, './/img[contains(@src, "lzd-img-global")]')
                details['image_url'] = image.get_attribute("src")

                self.data.append(details)
            except Exception:
                details['url'] = None
                details['image_url'] = None

        # self.driver.execute_script("window.scrollTo(0, 1000);")

        self.data = [dict(t) for t in {tuple(d.items())
                                       for d in self.data} if 'name' in dict(t)]

        return self.data

    def close_connection(self):
        self.driver.close()
