from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC
import time 
from selenium.webdriver.common.action_chains import ActionChains


from utils import *
import re
import os
import shutil
import sys

def resource_path(another_way):
    try:
        usual_way = sys._MEIPASS  # When in .exe, this code is executed, that enters temporary directory that is created automatically during runtime.
    except Exception:
        usual_way = os.path.dirname(__file__)  # When the code in run from python console, it runs through this exception.
    return os.path.join(usual_way, another_way)

def handle_table_date(date_str):

    try:
        date_str = date_str.upper()

        month_list = ["ENE", "FEB", "MAR", "ABR", "MAY", "JUN", "JUL", "AGO", "SEP", "OCT", "NOV", "DIC"]

        for month in month_list:
            if month in date_str:
                break
        
        date_field = date_str.split('/')
        date_field = list(filter(('').__ne__, date_field))

        for i, item in enumerate(date_field):
            if month in item:
                break
        
        day = int(date_field[i - 1])
        year = int(date_field[i + 1])

        month = month_list.index(month) + 1

        date_real = format((day), '02') + '/' + format((month), '02') + '/' + format((year), '04')
        return date_real
    
    except:
        return '**/**/****'

def Mongoconexion(database):
    MONGODB_HOST = '188.43.235.177'
    # MONGODB_HOST = 'localhost'
    MONGODB_PORT = '27017'
    MONGODB_TIMEOUT = 1000
    MONGODB_SOCKETTIMEOUT = 3000
    MONGODB_DATABASE = database
    MONGODB_USER = 'Admin-Aux'
    # MONGODB_USER = ''
    MONGODB_PASS = 'Admin-BMy-87FBhNy8-c5RzfF!'
    # MONGODB_PASS = ''
    URI_CONNECTION = 'mongodb://' + MONGODB_USER + ':' + MONGODB_PASS + '@' + MONGODB_HOST + ':' + MONGODB_PORT + '/admin'

    # URI_CONNECTION = "mongodb://localhost:27017/"
    

    client = pymongo.MongoClient(URI_CONNECTION, connectTimeoutMS=MONGODB_TIMEOUT,socketTimeoutMS=MONGODB_SOCKETTIMEOUT)
    
    return client[MONGODB_DATABASE]

def blink(number, interval = 1):
    for x in range(0, int(number / interval)):
        sys.stdout.write('\rPlease wait...{}s remaining...'.format(number - x * interval))
        time.sleep(0.5 * interval)
        sys.stdout.write('\r                               ')
        time.sleep(0.5 * interval)

# PREFIX = 'MainContent_rep_unidad_link_ir_'

PREFIX = 'MainContent_rep_unidad2_link_ir_'


if __name__ == '__main__':

    PATTERN = '^([0-2][1-9]|3[0-1])(.*)(19|20)[0-9][0-9]'

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # format 'download' folder
    pdf_files = os.listdir(os.path.join(dir_path, 'download'))
    pdf_files = [f for f in pdf_files if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        os.remove(os.path.join(dir_path, 'download', pdf_file))

    # reading MongoDB
    database = Mongoconexion('Actas')
    raw = database["raw"]

    # making check_list
    check_list = []
    cursor = raw.find({})

    # # make check_list of currennt database
    # try:
    #     for document in cursor:
    #         item = [document["juzgado"], document["fecha"]]
    #         if item not in check_list:
    #             check_list.append(item)
    # except:
    #     print('Mongodb error~~~')
    #     sys.exit()

    # run Google Chrome
    prefs = {"download.default_directory" : os.path.join(dir_path, 'download\\')}
    chromeOptions = webdriver.ChromeOptions()
    chromeOptions.add_experimental_option("prefs",prefs)
    chromedriver = os.path.join("driver", "chromedriver")
    driver = webdriver.Chrome(executable_path=resource_path(chromedriver), options=chromeOptions)

    # get Home page
    driver.get("https://boletinforaneojfca.stps.gob.mx/")

    boletin_num = driver.execute_script("return $('#MainContent_ddl_junta').children().length")

    print(boletin_num)

    while True:

        for k in range(23, boletin_num):

            driver.execute_script("$('#MainContent_ddl_junta').val({}).trigger('change')".format(k))

            time.sleep(10)

            for i in range(10000):

                driver.execute_script("$('#dataTables-example').DataTable().destroy();")

                time.sleep(0.5)

                button_id = PREFIX + str(i)

                try:
                    click_button = driver.find_element(By.ID,button_id)
                except:
                    print('Finished')
                    break
                
                tr_element = click_button.find_element(By.XPATH, '..//..')

                td_elements = tr_element.find_elements(By.TAG_NAME, 'td')

                juzgado = td_elements[0].text

                fecha = handle_table_date(td_elements[1].text)

                if fecha != '21/09/2021':
                    continue

                if [juzgado, fecha] not in check_list:

                    check_list.append([juzgado, fecha])
                    print([juzgado, fecha])

                    click_button = tr_element.find_element(By.TAG_NAME, "a")

                    ActionChains(driver).move_to_element(click_button).click(click_button).perform()

                    # find download button
                    btn_for_download = driver.find_element(By.ID, "MainContent_btn_generaboletin")

                    # click to start download
                    btn_for_download.click()

                    time.sleep(1)

                    # getting table data
                    table_data = []

                    try:
                        table_element = driver.find_element(By.ID,'table_expedientes')
                        rows = table_element.find_elements(By.TAG_NAME, "tr") # get all of the rows in the table
                        for row in rows:
                            # Get the columns (all the column 2)
                            cols = row.find_elements(By.TAG_NAME, "td")

                            if len(cols) != 2:
                                continue
                            
                            expendidate = cols[0].text
                            actor_demando = cols[1].text
                            table_data.append([expendidate, actor_demando])
                    except:
                        pass

                    try:
                        table_element = driver.find_element(By.ID,'table_convocatoria')
                        rows = table_element.find_elements(By.TAG_NAME, "tr") # get all of the rows in the table
                        for row in rows:
                            # Get the columns (all the column 2)
                            cols = row.find_elements(By.TAG_NAME, "td")

                            if len(cols) != 2:
                                continue

                            expendidate = cols[0].text
                            actor_demando = cols[1].text
                            table_data.append([expendidate, actor_demando])
                    except:
                        pass

                    try:
                        table_element = driver.find_element(By.ID,'table_amparos')
                        rows = table_element.find_elements(By.TAG_NAME, "tr") # get all of the rows in the table
                        for row in rows:
                            # Get the columns (all the column 2)
                            cols = row.find_elements(By.TAG_NAME, "td")

                            if len(cols) != 3:
                                continue

                            expendidate = [cols[0].text, cols[1].text]
                            actor_demando = cols[2].text
                            table_data.append([expendidate, actor_demando])
                    except:
                        pass


                    # back to home page
                    driver.back()

                    time.sleep(3)

                    # Get the contents from pdf file and handle exceptions
                    pdf_files = os.listdir(os.path.join(dir_path, 'download'))
                    pdf_files = [f for f in pdf_files if f.endswith('.pdf')]
                    for pdf_file in pdf_files:
                        try:
                            filename_list = pdf_file.split('_')
                            filename_list[1] = str(k + 16)
                            real_pdf_filename = '_'.join(filename_list)
                            print(real_pdf_filename)
                        except:
                            print(pdf_file)
                        pdf_path = os.path.join(dir_path, 'download', pdf_file)
                        # if not read_save_pyminer(pdf_path, raw):
                        #     if not read_save_fitz_with_table_data(pdf_path, table_data, raw):
                        #         shutil.copyfile(pdf_path, os.path.join("hard", pdf_file))
                        if not read_save_fitz_with_table_data(pdf_path, table_data):
                            shutil.copyfile(pdf_path, os.path.join("hard", pdf_file))
                        shutil.move(pdf_path, os.path.join("All", pdf_file))

        # waiting for 2hours
        blink(2 * 3600)
        # blink(2 * 10)
