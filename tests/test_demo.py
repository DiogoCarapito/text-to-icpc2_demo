# from text_to_icpc2_demo import main

from streamlit.testing.v1 import AppTest

pages = [
    "app.py",
    # "pages/2_Dataset.py",
    # "pages/3_Validação.py",
]

for each_page in pages:
    test = AppTest(each_page, default_timeout=120)
    test.run()
    assert not test.exception
