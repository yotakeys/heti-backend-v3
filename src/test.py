from service.tokopediaScraper import Tokopedia

tokopedia = Tokopedia("src/resources/chromedriver/chromedriver.exe")
produk = tokopedia.search("rollator")

print(produk)
