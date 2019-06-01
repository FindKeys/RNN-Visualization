import os
import zipfile
try:
    from urllib.request  import urlopen
except ImportError:
    from urllib2 import urlopen

def download_data(url):

	u = urlopen(url)
	data = u.read()
	u.close()
 
	path = './'

	with open(path+ 'mimic_viz.zip', "wb") as f :
		f.write(data)
	zip_ref = zipfile.ZipFile(path+'mimic_viz.zip', 'r')
	zip_ref.extractall(path)
	zip_ref.close()
	os.remove(path+'mimic_viz.zip')

if __name__ == "__main__":
	url = "https://www.dropbox.com/s/l9707bimzv6440c/ihm.zip?dl=1" 
	print('Downloading ...')
	download_data(url)
	print('Done.')