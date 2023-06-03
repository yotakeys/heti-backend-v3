from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import urllib.parse
import re
import time
import uuid


class Tokopedia:
    def __init__(self, chromedriver, headless=True) -> None:
        self.driver = self.setup(chromedriver, headless)
        self.data = []

    def setup(self, chromedriver, headless):
        opt = webdriver.ChromeOptions()
        opt.add_experimental_option('excludeSwitches', ['enable-logging'])
        if headless:
            opt.add_argument("--headless")
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
            name = detail_container.find_element(
                By.XPATH, ".//div[@data-testid='spnSRPProdName']").get_attribute("innerHTML")
            detail['name'] = name
        except Exception as e:
            ...

        # Price
        try:
            price = detail_container.find_element(
                By.XPATH, ".//div[@data-testid='spnSRPProdPrice']").get_attribute("innerHTML")
            price = float(re.sub('[^0-9]', '', price))
            detail['price'] = price
        except Exception as e:
            ...

        # Location
        try:
            location = detail_container.find_element(
                By.XPATH, ".//span[@data-testid='spnSRPProdTabShopLoc']").get_attribute("innerHTML")
            detail['location'] = location
        except Exception as e:
            ...

        # Rating
        try:
            rating = detail_container.find_element(By.XPATH, ".//*[contains(text(),'Terjual')]").find_element(
                By.XPATH, "preceding-sibling::span[2]").get_attribute("innerHTML")
            rating = float(rating)
            detail['rating'] = rating
        except Exception as e:
            detail['rating'] = 0

        # Sold
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
            detail['sold'] = 0

        return detail

    def search(self, cat):
        self.data = []

        url_safe_cat = urllib.parse.quote(cat)
        url = f"https://www.tokopedia.com/search?st=product&q={url_safe_cat}"
        # print(f'Scraping for category {cat}..')
        self.driver.get(url)

        # for i in range(2):
        #     time.sleep(1)
        containers = WebDriverWait(self.driver, 100).until(EC.presence_of_all_elements_located(
            (By.XPATH, "//div[@data-testid='master-product-card']")))

        for index, container in enumerate(containers):
            detail_container = container.find_element(By.TAG_NAME, "div").find_element(
                By.TAG_NAME, "div").find_elements(By.XPATH, "./div")[1].find_element(By.TAG_NAME, "a")
            details = self.get_details(detail_container, cat, index)
            try:
                links = container.find_element(
                    By.XPATH, './/a[contains(@href, "ta.tokopedia.com")]')
                url = links.get_attribute("href")
                encoded_uri = url.split("r=")[1]
                decoded_uri = urllib.parse.unquote(
                    encoded_uri).split("?")[0]
                details['url'] = decoded_uri

                image_container = container.find_element(By.TAG_NAME, "div").find_element(
                    By.TAG_NAME, "div").find_elements(By.XPATH, "./div")[0].find_element(By.TAG_NAME, "a").find_elements(
                        By.XPATH, "./div")[0].find_element(By.TAG_NAME, "img")
                image_url = image_container.get_attribute("src")
                image_url = [i for i in image_url.split("/")]
                img_url = f"{image_url[0]}//{image_url[2]}/{image_url[3]}/{image_url[6]}/{image_url[7]}/{image_url[8]}/{image_url[9]}/{image_url[10]}"

                details['image_url'] = img_url

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
