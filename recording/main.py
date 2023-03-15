from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC
import time 
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import re
import os
import shutil
import sys
import cv2
import pyautogui
import numpy as np

if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))

    target_url = 'https://www.protectedtext.com/'

    # target_url = 'https://translate.google.com/?sl=zh-CN&tl=en&op=translate'

    options = webdriver.ChromeOptions() 
    # options.add_argument("start-maximized")
    options.add_argument("--kiosk")
    options.add_argument("--app=http://www.google.com")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    

    
    chromedriver = os.path.join("driver", "chromedriver")
    # driver = webdriver.Chrome(executable_path=chromedriver)
    driver = webdriver.Chrome(options=options)
    # driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # get Home page
    driver.get(target_url)


    # Set the initial window size
    driver.set_window_size(1200, 800)

    # Set up the video writer
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter("output.mp4", fourcc, fps, (1200, 800))

    

    # Scroll down the webpage and capture frames
    SCROLL_AMOUNT = 30
    SCROLL_PAUSE_TIME = 0.01
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    x_pos, y_pos = (driver.execute_script("return [window.screenX, window.screenY];"))
    # Set up the screen capture region
    screen_region = (x_pos, y_pos, 1200, 800)

    while True:
        # Scroll down by a fixed amount
        driver.execute_script(f"window.scrollBy(0, {SCROLL_AMOUNT});")

        # # Wait for the page to settle
        time.sleep(SCROLL_PAUSE_TIME)

        # Capture a screenshot of the current frame
        img = pyautogui.screenshot(region=screen_region)

        # Convert the image to an OpenCV format and write to the video output
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video_out.write(frame)

        # cv2.imwrite('./', frame)

        # if old_frame is not None:

        #     difference = cv2.subtract(frame, old_frame)    
        #     result = not np.any(difference)

        #     if result:
        #         break
        
        # old_frame = frame.copy()

        # Calculate new scroll height and compare with last scroll height
        # new_height = driver.execute_script("return document.body.scrollHeight")
        current_scroll_position = driver.execute_script("return window.pageYOffset") + 800
        last_height = driver.execute_script("return document.body.scrollHeight")
        pass

        if current_scroll_position >= last_height:
            break
        # if new_height == last_height:
        #     break
        # last_height = new_height

    # Release the video writer and close the browser window
    video_out.release()
    driver.quit()