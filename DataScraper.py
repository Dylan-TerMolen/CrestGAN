import urllib.request
from bs4 import BeautifulSoup
import os
import random
import PIL.Image

base_logo_link = 'http://www.sportslogos.net/leagues/list_by_sport/5/Soccer/logos'
myPath = "/Users/Dylan/Desktop/ClubLogos"

def make_soup(url):
    page = urllib.request.urlopen(url)
    soup_data = BeautifulSoup(page, 'html.parser')
    return soup_data

def rename_dir_files_genericLabel(dirPath,genericLabel,newFileExtension):
    myDir = os.listdir(dirPath)
    for filename, i in zip(myDir, range(len(myDir))):
        newFileName = genericLabel + str(i) + newFileExtension
        os.rename(dirPath + '/' + filename, dirPath + '/' + newFileName)

def save_all_logos_from_link(url):
    mySoup = make_soup(url)
    imgCount = 0
    for img in mySoup.find_all('img'):
        imageLink = img.get('src')
        if 'content.sportslogos.net/logos' in imageLink:
            fileName = 'club_logo_' + str(random.randint(1,10000000000)) + '.jpg'
            imageFile = open(myPath + '/' + fileName,'wb')
            imageFile.write(urllib.request.urlopen(imageLink).read())
            imageFile.close()
            imgCount += 1
    print('Saved',imgCount,'images from link:',url)

def get_all_logos():
    base_soup = make_soup(base_logo_link)
    for a in base_soup.find_all('a'):
        if a.string is None and 'teams/list_by_league' in a.get('href') and 'FIFA' not in a.get('href'):
            logos_link = 'http://www.sportslogos.net/' + a.get('href')
            save_all_logos_from_link(logos_link)

def main():
    pass
if __name__ == '__main__':
    main()
