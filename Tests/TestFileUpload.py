import unittest
from app import app

class TestFileUpload(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_file_upload(self):
        with open('D:/PythonProject/Kyrsach_Savenkov_KPO/static/test_dataset.csv', 'rb') as file:
            data = {'file': (file, 'test_dataset.csv')}
            response = self.app.post('/calculate', data=data, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
