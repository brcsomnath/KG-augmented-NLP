import os
import sys
import preprocess

target_names = ['alt_atheism',
				'comp_graphics',
				'comp_os_ms_windows_m',
				'comp_sys_ibm_pc_hard',
				'comp_sys_mac_hardwar',
				'comp_windows_x',
				'misc_forsale',
				'rec_autos',
				'rec_motorcycles',
				'rec_sport_baseball',
				'rec_sport_hockey',
				'sci_crypt',
				'sci_electronics',
				'sci_med',
				'sci_space',
				'soc_religion_christi',
				'talk_politics_guns',
				'talk_politics_mideas',
				'talk_politics_misc',
				'talk_religion_misc']


data = []
label = []

def scrape_class(filepath, i):
	"""
	Used for reading the original dataset according to class
	The data is then preprocessed and saved as a duplicate copy for future use
	"""
	c = 0
	max_ = 0
	ch = 0
	for path, subdirs, files in os.walk(filepath):
		for filename in files:
			content = (preprocess.clean_str(preprocess.strip_newsgroup_quoting(preprocess.strip_newsgroup_header
				(preprocess.strip_newsgroup_footer(open(path+"/"+str(filename)).read())))))
			con = content.split(' ')
			if len(con)<300:
				file = open('Dataset/cleaned_data/'+target_names[i]+"/"+str(filename),"w")
				file.write(content)
				data.append(con)
				label.append(i)
				c += 1
				file.close()
	print 'Items in class '+target_names[i]+': %d'%(c)

	return c

def scrape_cleaned_data(filepath, i):
	"""
	Retrieves the saved dataset during training/testing process
	"""
	c = 0
	max_ = 0
	ch = 0
	for path, subdirs, files in os.walk(filepath):
		for filename in files:
			content = open(path+"/"+str(filename)).read().split(' ')
			while '' in content:
				content.remove('')
			if len(content)>0:
				data.append(content)
				label.append(i)
			c += 1
	print 'Items in class '+target_names[i]+': %d'%(c)
	return c


def retrieve_data_files():
	"""
	Utility function for scrape_cleaned_data() function
	"""
	count = 0
	for i in range(20):
		ct = scrape_cleaned_data('Dataset/cleaned_data/'+target_names[i], i)
		count += ct
	print 'Total items in the dataset %d.'%(count)
	return data, label


def setup():
	"""
	Utility function to setup cleaned dataset
	"""

	count = 0
	for i in range(20):
		ct = scrape_class('Dataset/cleaned_data/'+target_names[i], i)
		count += ct
	print 'Total items in the dataset %d.'%(count)

'''
if __name__ == '__main__':
	setup()
'''