# from text_to_icpc2_demo import main

from streamlit.testing.v1 import AppTest

pages = [
    "demo.py",
    "pages/2_Validation.py",
]


for each_page in pages:
    test = AppTest(each_page, default_timeout=30)
    test.run()
    assert not test.exception
