import json
from datetime import datetime
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from datetime import datetime
import re
import pymongo
import os
import shutil
import fitz  # this is pymupdf
from difflib import SequenceMatcher
import itertools



def upper_text(text):
    text = text.upper()
    text = text.replace('Á', 'A')
    text = text.replace('É', 'E')
    text = text.replace('Í', 'I')
    text = text.replace('Ó', 'O')
    text = text.replace('Ö', 'O')
    text = text.replace('Ú', 'U')
    text = text.replace('Ü', 'U')
    text = text.replace('Ñ', 'N')
    text = text.replace('  ', ' ')
    return text

def get_acuerdos(string, substring):

    candidate_string = ''
    acuerdos_string = ''

    try:
        if len(string) < len(substring):
            string, substring = substring, string

        content_list = string.split('\n')
        content_list = list(filter(('').__ne__, content_list))
        temp_string = ''
        coef_list = []
        for i in range(len(content_list)):
            temp_string += content_list[i]
            coef = similar(temp_string, substring)
            coef_list.append(coef)
            if len(temp_string) > 2 * len(substring):
                break
        # print(coef_list)
        max_index = coef_list.index(max(coef_list))
        actor_demando_string = ''.join(content_list[:max_index + 1])

        acuerdos_list = content_list[max_index + 1:]
        

        if len(acuerdos_list[-2].strip()) == 0:
            acuerdos_list = acuerdos_list[:-2]

        # print(acuerdos_list)

        acuerdos_string = '\n'.join(acuerdos_list)
        return acuerdos_string
    except Exception as e:
        # print(e)
        return ''

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

prefix_list = set(["TODA VEZ", "NO FUE", "SE NOTIFICA", "VISTO EL","NOTIFIQUESE", "SE TIENE POR", "AL NO","SE DA", 
                    "VISTA LA","DE CONFORMIDAD", "ARCHIVO GENERAL", 
                    "COMPARECENCIA", "PRESCRIPCION", "3 DIAS TERMINO", "CORRE TERMINO", 
                    "VISTO EL", "AUDIENCIA DE", "TERMINO ALEGATOS", "EMBARGO", "SE ORDENA", "A.G. CADUCIDAD", "EN EL", "EN TERMINOS", 
                    "SE NOTIFICA", "NOTIFIQUESE", "SE TIENE POR", "SE DA", "ARCHIVO GENERAL", "COMPARECENCIA", 
                    "PRESCRIPCION", "3 DIAS TERMINO", "CORRE TERMINO", 
                    "SOBRESEE", "ALEGATOS", "POR RECIBIDO OFICIO", "SE GIRA", 
                    "SE EMITE", "LAUDO", "ACUERDO EN", "RADICACION", "ACUERDO DE", "RESOLUCION INCIDENTAL", "SE REGULARIZA",
                    "SE \nREGULARIZA"])

def find_common_string(string1, string2):
    match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
    return string1[match.a: match.a + match.size]

def convert_pdf_to_txt(path):
    resource_manager = PDFResourceManager()
    device = None
    try:
        with StringIO() as string_writer, open(path, 'rb') as pdf_file:
            device = TextConverter(resource_manager, string_writer, laparams=LAParams(char_margin = 20))
            interpreter = PDFPageInterpreter(resource_manager, device)

            for page in PDFPage.get_pages(pdf_file):
                interpreter.process_page(page)

            pdf_text = string_writer.getvalue()
    finally:
        if device:
            device.close()
    return pdf_text

def handle_date(date_str):

    try:
        date_str = date_str.replace('de', '')
        date_str = date_str.upper()

        month_list = ["ENE", "FEB", "MAR", "ABR", "MAY", "JUN", "JUL", "AGO", "SEP", "OCT", "NOV", "DIC"]

        for month in month_list:
            if month in date_str:
                break
        
        date_field = date_str.split(' ')
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

def find_vs(sub_string):
    index = sub_string.find("Vs")
    if index == -1:
        index = sub_string.find("vs")
        if index == -1:
            index = sub_string.find("VS")
    return index

def find_acuerdos(sub_string):
    prefix_list = ["NO FUE", "EN EL", "SE NOTIFICA", "VISTO EL","NOTIFIQUESE", "SE TIENE POR", "AL NO","SE DA", "VISTA LA","DE CONFORMIDAD", "ARCHIVO GENERAL", 
                    "COMPARECENCIA", "PRESCRIPCION", "3 DIAS TERMINO", "CORRE TERMINO", 
                    "VISTO EL", "AUDIENCIA DE", "TERMINO ALEGATOS", "EMBARGO", "SE ORDENA", "A.G. CADUCIDAD",  "EN TERMINOS"]
    for string in prefix_list:
        index = sub_string.find(string)
        if index != -1:
            break
    return index

def read_save_pyminer(pdf_filename, mongdb_name = False):

    try:
        # convert pdf to text
        text = convert_pdf_to_txt(pdf_filename)
        text_list = text.split('\n')
        
        # check page numbers and title
        for sub_text in text_list:
            if 'g i n a' in sub_text:
                text = text.replace(sub_text, '')
            if 'Junta Especial' in sub_text:
                JUZGADO = sub_text
                text = text.replace(sub_text, '')
                JUZGADO = JUZGADO.strip()
            if 'Num' in sub_text and 'de' in sub_text:
                if len(sub_text) > 40:
                    # print((sub_text))

                    sub_text_list = sub_text.split(',')
                    if len(sub_text_list) < 2:
                        continue
                    ENTIDAD = sub_text_list[0]
                    FECHA = sub_text_list[1]
                    FECHA = handle_date(FECHA)

                    text = text.replace(sub_text, '')

        # remove multiple blank enter
        text = text.replace('\n\n\n', '\n')

        # modify ENTIDAD
        # ENTIDAD = ENTIDAD.replace('Guadalajara', 'Guadalaiara')
        
        # replace letters
        if text.find("Expedientes") != -1:
            content = text[text.find("Expedientes"):]
        else:
            content = text
        content = content.upper()
        content = content.replace('Á', 'A')
        content = content.replace('É', 'E')
        content = content.replace('Í', 'I')
        content = content.replace('Ó', 'O')
        content = content.replace('Ö', 'O')
        content = content.replace('Ú', 'U')
        content = content.replace('Ü', 'U')
        # content = content.replace('Ñ', 'N')
        content = content.replace('  ', ' ')
        content = content.replace('“', '"')
        content = content.replace('”', '"')
        
        # print(content)
        old_content = content
        adding = 0
        index_list = []
        i = 0
        gropu = []

        while True:
            i += 1
            order_string = '\n' + str(i) + '.'
            try:
                num_index = content.find(order_string)
                if num_index == -1:
                    num_index = content.find(str(i) + '.')
                    if num_index + adding < index_list[-1]:
                        break
                    pp = content[num_index-3 : num_index-1]
                    if pp == '20' or pp == '19':
                        break
                index = num_index + adding
                if index - adding == -1:

                    order_string = '\n1.'
                    new_content = content[index_list[-1]:]
                    index = new_content.find(order_string)
                    if index != -1:
                        gropu.append(i)
                        i = 1
                        content = new_content[index:]
                        index = index + index_list[-1]
                        adding = index
                    else:
                        break
                index_list.append(index)
            except:
                break

        index_list.append(-1)
        content = old_content
        outputs = []
        
        try:
            for i in range(len(index_list)-1):
                # print(i + 1)
                save_dict = {}
                sub_strings = []

                sub_contents = content[index_list[i] : index_list[i + 1]]

                sub_string_list = sub_contents.split('\n\n')

                new_sub_string_list = []
                for item in sub_string_list:
                    if item.startswith('SECRETARIO DE'):
                        continue
                    else:
                        new_sub_string_list.append(item)
                sub_string_list = new_sub_string_list

                new_sub_string_list = []
                for item in sub_string_list:
                    item = item.strip()
                    if len(item) > 0:
                        new_sub_string_list.append(item)

                if len(new_sub_string_list) > 3:
                    for i, sub_string in enumerate(new_sub_string_list):
                        breaking = False
                        for prefix in prefix_list:
                            if prefix in sub_string:
                                breaking = True
                                break
                        if breaking:
                            break
                    if not breaking:
                        i = 2
                    new_sub_string_list[1] = ' '.join(new_sub_string_list[1:i])
                    new_sub_string_list[2] = ' '.join(new_sub_string_list[i:])
                    # print()

                expediente = new_sub_string_list[0]
                expediente = expediente.replace('\n', '')
                expediente = expediente.split('.')[-1]
                expediente = expediente.strip()

                versus_index = find_vs(new_sub_string_list[1])
                sub_strings.append(new_sub_string_list[1][:versus_index])
                sub_strings.append(new_sub_string_list[1][versus_index + 2:])
                sub_strings.append(new_sub_string_list[2])

                for j, sub_string in enumerate(sub_strings):
                    sub_string = sub_string.split('\n')
                    sub_string = list(filter(('').__ne__, sub_string))
                    
                    sub_string = ''.join(sub_string)
                    sub_string = sub_string.strip()
                    sub_string = sub_string.replace('  ', ' ')
                    sub_strings[j] = sub_string

                save_dict["actor"] = sub_strings[0]
                save_dict["demandado"] = sub_strings[1]
                save_dict["entidad"] = ENTIDAD
                save_dict["expediente"] = expediente
                save_dict["fecha"] = FECHA
                save_dict["fuero"] = 'COMUN'
                save_dict["juzgado"] = JUZGADO
                save_dict["tipo"] = "LABORAL"
                save_dict["acuerdos"] = sub_strings[2]
                save_dict["monto"] = ''
                save_dict["fecha_presentacion"] = ''
                save_dict["actos_reclamados"] = ''
                save_dict["actos_reclamados_especificos"] = ''
                save_dict["naturaleza_procedimiento"] = ''
                save_dict["prestacion_demandada"] = ''
                save_dict["organo_jurisdiccional_origen"] = ''
                save_dict["expediente_origen"] = ''
                save_dict["materia"] = "LABORAL"  #expediente
                save_dict["submateria"] = ''
                save_dict["fecha_sentencia"] = ''
                save_dict["sentido_sentencia"] = ''
                save_dict["resoluciones"] = ''
                save_dict["origen"] = 'JUNTA FEDERAL DE CONCILIACION Y ARBITRAJE'

                save_dict["fecha_insercion"] = datetime.utcnow()
                # save_dict["fecha_insercion"] = datetime.now().strftime("%m/%d/%Y")
                save_dict["fecha_tecnica"] = datetime.utcnow().year
                # save_dict["fecha_tecnica"] = datetime.now().strftime("%Y")
                outputs.append(save_dict)

            if len(outputs) == 0:
                return False

            for output in outputs:
                # print(output)
                if mongdb_name is not None:
                    mongdb_name.insert_one(output)
            return True
        except Exception as e:
            # print(e)
            return False
    except Exception as e:
        # print(e)
        return False

def read_save_fitz(pdf_filename, mongdb_name = False):

    try:
            
        common_title = []

        with fitz.open(pdf_filename) as doc:
            text = ""
            for i, page in enumerate(doc):
                sub_text = page.get_text()
                text += sub_text

                if len(common_title) < 2:
                    common_title.append(sub_text)
            
        page_num = i + 1

        # print(text)

        title = ''
        try:
            common_title_string_1 = find_common_string(common_title[0][:100], common_title[1][:100])
            common_title_string_2 = common_title[0][:common_title[0].find('Num')]
            if len(common_title_string_1) > len(common_title_string_2):
                common_title_string = common_title_string_1
            else:
                common_title_string = common_title_string_2
            common_title_list = common_title_string.split('   ')

            if len(common_title_list) >= 2 :
                for item in common_title_list:
                    if len(item) == 0:
                        continue
                    if len(title) == 0:
                        title = item
            
            JUZGADO = text[:text.find(title)]
            JUZGADO = JUZGADO.strip()
            footer = JUZGADO
            if JUZGADO == '':
                JUZGADO = text[:text.find('g i n a')]
            JUZGADO = JUZGADO.split('\n')
            for item in JUZGADO:
                item = item.strip()
                if len(item) > 10:
                    break
            JUZGADO = item

            footer = footer[footer.find(JUZGADO):].replace(JUZGADO, '')
            # if footer == '':
            #     footer = 

            for i in range(page_num):
                new_footer = footer.replace('1', str(i + 1))
                text = text.replace(new_footer, '')
            text = text.replace(JUZGADO.upper(), '')
            title_list = title.split('\n') 

            if title == '':
                raise ValueError
        except:
            title = text[:text.find("Expedientes")]
            if len(title) > 100:
                title = text[:text.find("Num.")]
            text = text.replace(title, '')
            split_title = title.split('\n')
            title_list = []
            for item in split_title:
                item =item.strip()
                if len(item) == 0:
                    continue
                title_list.append(item)
            
            if page_num == 1:
                JUZGADO = title_list[0]
            title_list = [title_list[-1]]
            common_title_string = title_list[0]

            if page_num > 1:
                footer = 'P á g i n a  1'

                for i in range(page_num):
                    new_footer = footer.replace('1', str(i + 1))
                    ' | 6'
                    text = text.replace(new_footer, '')
                    text = text.replace(' | ' + str(i + 1), '')
                text = text.replace(JUZGADO.upper(), '')
        
        for item in title_list:
            item = item.strip()
            if len(item) == 0:
                continue
            if len(item) < 10:
                continue
            
            item = item.split(',')
            if len(item) < 2:
                continue
            ENTIDAD = item[0]
            FECHA = item[1]

        for i in range(page_num):
            footer = title.replace('g i n a  1', 'g i n a  ' + str(i + 1))
            text = text.replace(footer, '')

        # modify ENTIDAD
        # ENTIDAD = ENTIDAD.replace('Guadalajara', 'Guadalaiara')

        text_list = text.split('\n')
        new_text_list = []
        for text in text_list:
            if 'g i n a' in text:
                continue 
            if similar(FECHA, text) > 0.7:
                continue
            if FECHA in text:
                continue
            if 'Num.' in text:
                continue 
            new_text_list.append(text)
        text = '\n'.join(new_text_list)
        
        # modify FECHA
        FECHA = handle_date(FECHA)

        try:
            text = text.replace(common_title_string, '')
        except:
            pass
        
        if text.find("Expedientes") != -1:
            content = text[text.find("Expedientes"):]
        else:
            content = text
        
        if True:
            content = content.upper()
            content = content.replace('Á', 'A')
            content = content.replace('É', 'E')
            content = content.replace('Í', 'I')
            content = content.replace('Ó', 'O')
            content = content.replace('Ö', 'O')
            content = content.replace('Ú', 'U')
            content = content.replace('Ü', 'U')
            content = content.replace('Ñ', 'N')
            content = content.replace('  ', ' ')
            content = content.replace(JUZGADO.upper(), '')

        old_content = content
        adding = 0
        index_list = []
        i = 0
        gropu = []

        while True:
            i += 1
            order_string = '\n' + str(i) + '.'
            try:
                num_index = content.find(order_string)
                if num_index == -1:
                    num_index = content.find(str(i) + '.')
                    if num_index != -1 and num_index + adding < index_list[-1]:
                        break
                index = num_index + adding
                if index - adding == -1:

                    order_string = '\n1.'
                    new_content = content[index_list[-1]:]
                    index = new_content.find(order_string)
                    if index != -1:
                        gropu.append(i)
                        i = 1
                        content = new_content[index:]
                        index = index + index_list[-1]
                        adding = index
                    else:
                        break
                index_list.append(index)
            except:
                break

        new_index_list = [index_list[0]]
        for i in range(1, len(index_list)):
            if index_list[i] - index_list[i - 1] < 10:
                continue
            new_index_list.append(index_list[i])
        index_list = new_index_list
        index_list.append(-1)
        content = old_content
        outputs = []

        try:
            for i in range(len(index_list)-1):
                print(i + 1)
                print()
                save_dict = {}
                sub_contents = content[index_list[i] : index_list[i + 1]]
                if len(gropu) != 0:
                    for number in gropu:
                        if i + 1 >= number:
                            i = i - number + 1
                            break

                sub_contents = sub_contents.replace(str(i + 1) + '.', '')

                if len(sub_contents) == 0:
                    continue
                
                prefix = False
                available_prefix_dict = {}
                for _prefix in prefix_list:
                    _prefix = '\n' + _prefix
                    if _prefix in sub_contents:
                        # print(prefix)
                        prefix = _prefix.replace('\n', '')
                        available_prefix_dict[prefix] = sub_contents.find(_prefix)

                        sub_contents = sub_contents.replace(_prefix, prefix)
                        # .append(sub_contents.find(prefix))
                        # break
                
                sub_string = sub_contents.split('\n')
                sub_string = list(filter(('').__ne__, sub_string))
                expediente = sub_string[0].strip()

                # sub_contents = sub_contents.replace('\n', '')
                sub_contents = sub_contents.replace(expediente, '')
                adding = sub_string[1].strip()

                if 'AMPARO' in sub_contents:
                    for i, item in enumerate(sub_string):
                        if 'AMPARO' in item:
                            if i == len(sub_string) - 1:
                                break
                            adding = sub_string[i + 1]
                            try:
                                int(adding[-2:])
                                for j in range(1, i + 2):
                                    expediente += ' ' + sub_string[j]
                                    sub_contents = sub_contents.replace(sub_string[j], '')
                                expediente = expediente.replace('  ', ' ')
                                break
                            except:
                                break

                sub_contents = sub_contents.replace('“', ' ')
                sub_contents = sub_contents.replace('”', ' ')
                

                versus_index = find_vs(sub_contents)
                if len(available_prefix_dict) == 0:
                    acuerdos_index = find_acuerdos(sub_contents)
                else:
                    prefix = min(available_prefix_dict, key=available_prefix_dict.get)
                    acuerdos_index = sub_contents.find(prefix)

                sub_strings = []
                

                sub_strings.append(sub_contents[:versus_index])
                sub_strings.append(sub_contents[versus_index + 2:acuerdos_index])
                sub_strings.append(sub_contents[acuerdos_index:])

                for j, sub_string in enumerate(sub_strings):
                    if j == 2:
                        if '\nREPRESENTANTE' in sub_string:
                            sub_string = sub_string[:sub_string.find('REPRESENTANTE DE')]
                            try:
                                sub_string = sub_string[:sub_string.rindex('.')]
                                sub_string += '.'
                            except:
                                pass
                        if '\nSECRETARIO DE' in sub_string:
                            sub_string = sub_string[:sub_string.find('SECRETARIO DE')]
                    
                    sub_string = sub_string.split('\n')
                    sub_string = list(filter(('').__ne__, sub_string))

                    sub_string = ''.join(sub_string)
                    sub_string = sub_string.strip()
                    sub_strings[j] = sub_string

                # current_time = datetime.now()

                save_dict["actor"] = sub_strings[0]
                save_dict["demandado"] = sub_strings[1]
                save_dict["entidad"] = ENTIDAD
                save_dict["expediente"] = expediente
                save_dict["fecha"] = FECHA
                save_dict["fuero"] = 'COMUN'
                save_dict["juzgado"] = JUZGADO
                save_dict["tipo"] = "LABORAL"    #expediente
                save_dict["acuerdos"] = sub_strings[2]
                save_dict["monto"] = ''
                save_dict["fecha_presentacion"] = ''
                save_dict["actos_reclamados"] = ''
                save_dict["actos_reclamados_especificos"] = ''
                save_dict["naturaleza_procedimiento"] = ''
                save_dict["prestacion_demandada"] = ''
                save_dict["organo_jurisdiccional_origen"] = ''
                save_dict["expediente_origen"] = ''
                save_dict["materia"] = "LABORAL"  #expediente
                save_dict["submateria"] = ''
                save_dict["fecha_sentencia"] = ''
                save_dict["sentido_sentencia"] = ''
                save_dict["resoluciones"] = ''
                save_dict["origen"] = 'JUNTA FEDERAL DE CONCILIACION Y ARBITRAJE'
                save_dict["fecha_insercion"] = datetime.utcnow()
                # save_dict["fecha_tecnica"] = datetime.strptime(FECHA, "%m/%d/%y")

                outputs.append(save_dict)
            
            if len(outputs) == 0:
                return False

            for output in outputs:
                print(output)
                # if mongdb_name is not None:
                #     mongdb_name.insert_one(output)
            return True
        except Exception as e:
            print(e)
            return False
    except Exception as e:
        print(e)
        return False

def read_save_fitz_with_table_data(pdf_filename, table_data, mongdb_name = False):

    try:
            
        common_title = []

        with fitz.open(pdf_filename) as doc:
            text = ""
            for i, page in enumerate(doc):
                sub_text = page.get_text()
                text += sub_text

                if len(common_title) < 2:
                    common_title.append(sub_text)
            
        page_num = i + 1

        # print(text)

        title = ''
        try:
            common_title_string_1 = find_common_string(common_title[0][:100], common_title[1][:100])
            common_title_string_2 = common_title[0][:common_title[0].find('Num')]
            if len(common_title_string_1) > len(common_title_string_2):
                common_title_string = common_title_string_1
            else:
                common_title_string = common_title_string_2
            common_title_list = common_title_string.split('   ')

            if len(common_title_list) >= 2 :
                for item in common_title_list:
                    if len(item) == 0:
                        continue
                    if len(title) == 0:
                        title = item
            
            JUZGADO = text[:text.find(title)]
            JUZGADO = JUZGADO.strip()
            footer = JUZGADO
            if JUZGADO == '':
                JUZGADO = text[:text.find('g i n a')]
            JUZGADO = JUZGADO.split('\n')
            for item in JUZGADO:
                item = item.strip()
                if len(item) > 10:
                    break
            JUZGADO = item

            footer = footer[footer.find(JUZGADO):].replace(JUZGADO, '')
            # if footer == '':
            #     footer = 

            for i in range(page_num):
                new_footer = footer.replace('1', str(i + 1))
                text = text.replace(new_footer, '')
            text = text.replace(JUZGADO.upper(), '')
            title_list = title.split('\n') 

            if title == '':
                raise ValueError
        except:
            title = text[:text.find("Expedientes")]
            if len(title) > 100:
                title = text[:text.find("Num.")]
            text = text.replace(title, '')
            split_title = title.split('\n')
            title_list = []
            for item in split_title:
                item =item.strip()
                if len(item) == 0:
                    continue
                title_list.append(item)
            
            if page_num == 1:
                JUZGADO = title_list[0]
            title_list = [title_list[-1]]
            common_title_string = title_list[0]

            if page_num > 1:
                footer = 'P á g i n a  1'

                for i in range(page_num):
                    new_footer = footer.replace('1', str(i + 1))
                    ' | 6'
                    text = text.replace(new_footer, '')
                    text = text.replace(' | ' + str(i + 1), '')
                text = text.replace(JUZGADO.upper(), '')
        
        for item in title_list:
            item = item.strip()
            if len(item) == 0:
                continue
            if len(item) < 10:
                continue
            
            item = item.split(',')
            if len(item) < 2:
                continue
            ENTIDAD = item[0]
            FECHA = item[1]

        for i in range(page_num):
            footer = title.replace('g i n a  1', 'g i n a  ' + str(i + 1))
            text = text.replace(footer, '')

        # modify ENTIDAD
        # ENTIDAD = ENTIDAD.replace('Guadalajara', 'Guadalaiara')

        text_list = text.split('\n')
        new_text_list = []
        for text in text_list:
            if 'g i n a' in text:
                continue 
            if similar(FECHA, text) > 0.7:
                continue
            if FECHA in text:
                continue
            if 'Num.' in text:
                continue 
            new_text_list.append(text)
        text = '\n'.join(new_text_list)
        
        # modify FECHA
        FECHA = handle_date(FECHA)

        try:
            text = text.replace(common_title_string, '')
        except:
            pass
        
        if text.find("Expedientes") != -1:
            content = text[text.find("Expedientes"):]
        else:
            content = text
        
        
        if True:
            content = upper_text(content)
            content = content.replace(JUZGADO.upper(), '')

        index_list = []

        normal_table = []
        amparos_table = []
        for data in table_data:
            if len(data[0][0]) == 1:
                normal_table.append(data)
            else:
                amparos_table.append(data)

        if len(normal_table) != 0:
            normal_table = list(table_data for table_data,_ in itertools.groupby(normal_table))
            for i, data in enumerate(normal_table):
                temp_expendidate, temp_actor_demando = data
                temp_actor_demando = upper_text(temp_actor_demando)

                expendidate_index = content.find(temp_expendidate)
                if expendidate_index != -1:
                    index_list.append(expendidate_index)
                
                normal_table[i] = [temp_expendidate, temp_actor_demando]

            for i in range(len(index_list)):
                for j in range(i, len(index_list)):
                    if index_list[i] > index_list[j]:
                        normal_table[i], normal_table[j] = normal_table[j], normal_table[i]
                        index_list[i], index_list[j] = index_list[j], index_list[i] 

            for k in range(len(index_list) - 1):
                if index_list[k + 1] - index_list[k] < 5:
                    temp_expendidate = normal_table[k + 1][0]
                    new_index = [m.start() for m in re.finditer(temp_expendidate, content)][-1]
                    index_list[k + 1] = new_index

                    for i in range(len(index_list)):
                        for j in range(i, len(index_list)):
                            if index_list[i] > index_list[j]:
                                normal_table[i], normal_table[j] = normal_table[j], normal_table[i]
                                index_list[i], index_list[j] = index_list[j], index_list[i] 
        if len(amparos_table) != 0:
            for i, data in enumerate(amparos_table):
                temp_expendidate_list, temp_actor_demando = data
                temp_actor_demando = upper_text(temp_actor_demando)

                temp_expendidate = 'Expediente Laboral: ' + temp_expendidate_list[0] + ' Amparo: ' + temp_expendidate_list[1]
                amparos_table[i] = [temp_expendidate, temp_actor_demando]
            
            adding_list = [m.start() for m in re.finditer('EXPEDIENTE LABORAL: ', content)]

            if len(adding_list) > len(amparos_table):
                adding_list = adding_list[-len(amparos_table):]
            index_list += adding_list

        table_data = normal_table + amparos_table

        for i in range(len(index_list)):
            for j in range(i, len(index_list)):
                if index_list[i] > index_list[j]:
                    table_data[i], table_data[j] = table_data[j], table_data[i]
                    index_list[i], index_list[j] = index_list[j], index_list[i] 
        index_list.append(-1)

        outputs = []

        try:
            for i in range(len(index_list)-1):
                print(i + 1)
                print()
                save_dict = {}
                sub_contents = content[index_list[i] : index_list[i + 1]]

                if len(sub_contents) == 0:
                    continue
                
                # get expediente and actor_demando
                expediente = table_data[i][0]
                temp_actor_demando = table_data[i][1]

                sub_contents = sub_contents.replace(expediente, '')

                versus_index = find_vs(temp_actor_demando)

                sub_strings = []
                
                sub_strings.append(temp_actor_demando[:versus_index])
                sub_strings.append(temp_actor_demando[versus_index + 2:])

                acuerdos = get_acuerdos(sub_contents, temp_actor_demando)

                sub_strings.append(acuerdos)

                for j, sub_string in enumerate(sub_strings):
                    if j == 2:
                        if '\nREPRESENTANTE' in sub_string:
                            sub_string = sub_string[:sub_string.find('\nREPRESENTANTE DE')]
                        if '\nSECRETARIO DE' in sub_string:
                            sub_string = sub_string[:sub_string.find('\nSECRETARIO DE')]
                        if '\nCONVOCATORIA' in sub_string:
                            sub_string = sub_string[:sub_string.find('\nCONVOCATORIA')]

                    sub_string = sub_string.split('\n')
                    sub_string = list(filter(('').__ne__, sub_string))

                    sub_string = ''.join(sub_string)
                    sub_string = sub_string.strip()
                    sub_strings[j] = sub_string

                # current_time = datetime.now()

                save_dict["actor"] = sub_strings[0]
                save_dict["demandado"] = sub_strings[1]
                save_dict["entidad"] = ENTIDAD
                save_dict["expediente"] = expediente
                save_dict["fecha"] = FECHA
                save_dict["fuero"] = 'COMUN'
                save_dict["juzgado"] = JUZGADO
                save_dict["tipo"] = "LABORAL"    #expediente
                save_dict["acuerdos"] = sub_strings[2]
                save_dict["monto"] = ''
                save_dict["fecha_presentacion"] = ''
                save_dict["actos_reclamados"] = ''
                save_dict["actos_reclamados_especificos"] = ''
                save_dict["naturaleza_procedimiento"] = ''
                save_dict["prestacion_demandada"] = ''
                save_dict["organo_jurisdiccional_origen"] = ''
                save_dict["expediente_origen"] = ''
                save_dict["materia"] = "LABORAL"  #expediente
                save_dict["submateria"] = ''
                save_dict["fecha_sentencia"] = ''
                save_dict["sentido_sentencia"] = ''
                save_dict["resoluciones"] = ''
                save_dict["origen"] = 'JUNTA FEDERAL DE CONCILIACION Y ARBITRAJE'
                save_dict["fecha_insercion"] = datetime.utcnow()
                # save_dict["fecha_tecnica"] = datetime.strptime(FECHA, "%m/%d/%y")
                
                print(save_dict)

                outputs.append(save_dict)
            
            if len(outputs) == 0:
                return False

            # for output in outputs:
                # print(output)
                # if mongdb_name is not None:
                #     mongdb_name.insert_one(output)
            return True
        except Exception as e:
            print(e)
            return False
    except Exception as e:
        print(e)
        return False



if __name__ == '__main__':

    client = pymongo.MongoClient("mongodb://localhost:27017/")
    database = client["boletin"]
    raw = database["raw"]
    
    pdf_path = "hard\\boletin_17_2020-09-28.pdf"

    # print(read_save_pyminer(pdf_path, raw))

    print(read_save_fitz(pdf_path, raw))
