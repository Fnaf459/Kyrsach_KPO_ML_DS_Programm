import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

class TestFileUploadSelenium(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Chrome()  # Используйте свой драйвер, например, ChromeDriver

    def tearDown(self):
        self.driver.quit()

    def test_file_upload(self):
        self.driver.get("http://localhost:5000/")  # Замените URL на свой, если необходимо

        # Найдем элемент загрузки файла и прикрепим файл
        file_input = self.driver.find_element(By.ID, "fileInput")
        file_input.send_keys("D:/PythonProject/Kyrsach_Savenkov_KPO/static/test_dataset.csv")

        # Найдем кнопку отправки формы и кликнем на нее
        submit_button = self.driver.find_element(By.XPATH, "//button[@type='submit']")
        submit_button.click()

        # Добавьте паузу для того, чтобы веб-страница успела обработать запрос
        time.sleep(2)

        # Проверим, что результативность отображается
        result_div = self.driver.find_element(By.XPATH, "//td")
        self.assertTrue(result_div.is_displayed())

if __name__ == '__main__':
    unittest.main()
