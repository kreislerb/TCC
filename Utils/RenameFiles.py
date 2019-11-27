# Pythono3 code to rename multiple
# files in a directory or folder

# importing os module
import os


# Function to rename multiple files
def main():

    folderIn = 'database'
    folderOut = 'database2'
    for filename in os.listdir(folderIn):

        filenameDividido = filename.split('.')
        #print(filenameDividido)

        filenameRenamed = filenameDividido[0] + '-' + filenameDividido[1]
        print(filenameRenamed)
        src = folderIn + '/' + filename
        dst = folderOut + '/'+ filenameRenamed

        # rename() function will
        # rename all the files
        os.rename(src, dst)


# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()