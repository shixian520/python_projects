# STEP 1
# import libraries
import fitz
import io
from PIL import Image
import glob
import os
  
# STEP 2
# file path you want to extract images from

pdf_path = 'C:\\Users\\Man\\Downloads\\Unsorted FakeReal'
pdf_path = glob.glob(pdf_path + "**/*.pdf", recursive = True)
# file = "temp/party_8277658_gdg_117090923334749962.pdf"
  
# open the file
for pdf_file in pdf_path:
    
    print(pdf_file)

    filename = os.path.basename(pdf_file)
    filename = filename.split('.')[0]

    pdf_file = fitz.open(pdf_file)
    # STEP 3
    # iterate over PDF pages
    for page_index in range(len(pdf_file)):
        
        # get the page itself
        page = pdf_file[page_index]
        image_list = page.getImageList()
        
        # printing number of images found in this page
        if image_list:
            print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
        else:
            print("[!] No images found on page", page_index)
        for image_index, img in enumerate(page.getImageList(), start=1):
            
            # get the XREF of the image
            xref = img[0]
            
            # extract the image bytes
            base_image = pdf_file.extractImage(xref)
            image_bytes = base_image["image"]
            
            # get the image extension
            image_ext = base_image["ext"]
            if image_ext in ['jpg', 'jpeg', 'png']:
                image_ext = 'jpg'

                image = Image.open(io.BytesIO(image_bytes))
                # save it to local disk
                save_path = 'output/' + filename + '_' + str(image_index) + '.' + image_ext
                image.save(open(save_path, "wb"))

                # print('Finished...')
            else:
                # image_ext = 'jpeg'
                pass

                image = Image.open(io.BytesIO(image_bytes))
                # save it to local disk
                save_path = 'output/' + filename + '_' + str(image_index) + '.' + image_ext
                image.save(open(save_path, "wb"))

                print('Finished...')