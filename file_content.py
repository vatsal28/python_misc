#!/usr/bin/env python3



import os

import logging

import csv

import numpy as np

import pandas as pd



from gensim.summarization import summarize



path_to_directory = os.path.join(os.path.expanduser('~'), 'Desktop')+"/Escalations"



file_content = []

sender_name = []

#std_email_delimiter = "-- \n"
#std_email_delimiter2 = "--\n"
#outlook_delim = "-----Original Message-----"
#outlook_delim2 = "________________________________"
#os_mail_start = "On "
#os_mail_end = " wrote:\n"
#outlook_failsafe = "From: "
#os_sent_line = "Sent from my iPhone"
#os_sent_line2 = "Sent from my iPad"
#outlook_delim3 = "Thanks," 
#outlook_delim4 = "Thanks & Regards,"
#outlook_delim5 = "Regards,"


for filename in os.listdir(path_to_directory):

	if filename.endswith(".txt"):

		#f=open(filename)

		f=open(os.path.join(os.path.expanduser('~'), 'Desktop')+"/Escalations/"+filename)

		lines=f.read()
		#lines = lines.split(std_email_delimiter, 1)[0]
		#lines = lines.split(std_email_delimiter2, 1)[0]
		#lines = lines.split(outlook_delim, 1)[0]
		#lines = lines.split(outlook_delim2, 1)[0]
		#lines = lines.split(outlook_delim3, 1)[0]
		#lines = lines.split(outlook_delim4, 1)[0]
		#lines = lines.split(outlook_delim5, 1)[0]

		#lines = lines.split(outlook_failsafe, 1)[0]
		#lines = lines.split(os_sent_line, 1)[0]
		#lines = lines.split(os_sent_line2, 1)[0]
		lines=(lines,"utf-8")
		lines =' '.join(lines)
		f.close()

		lines=lines.replace("\n"," ")

		lines=lines.replace("\xa0","")

		#print(filename)

		#f.close()

		if(len(lines)<=500):

			less=summarize(lines,ratio=0.8)

			less=less.replace("\n"," ")

			less=less.replace("\xa0","")

			file_content.append(less)

			sender_name.append(filename)		

		else:

			more=summarize(lines,ratio=0.5)

			more=more.replace("\n"," ")

			more=more.replace("\xa0","")

			file_content.append(more)

			sender_name.append(filename)



#sender_name

#file_content

#output_file = open("output_mail_content.csv","w")



#for i in range(len(sender_name)):

#	output_file.write("{} {}\n".format(sender_name[i],file_content[i]))



#output_file.close()

			

#xarray = np.array(sender_name)

#yarray = np.array(file_content)

blah_list = pd.DataFrame({'Sender_name':sender_name,'Content':file_content})

blah_list.to_csv("~/Desktop/Escalations/Escal_details.csv",index=False)

print("Mail summarization and sender details extraction done")

#data = np.array([xarray, yarray])

#data = data.T

#numpy.savetxt("foo.csv", data, delimiter=",")

#numpy.savetxt("foo.csv", data)

#print(file_content)

#out = csv.writer(open("mail_content_new_final.csv",'w'))

#print(type(data))

#data = zip(sender_name,file_content)

#for row in data:

#	out.writerow(data)

#print (file_content)		