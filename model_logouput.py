import os

class model_logouput:

    output_path= ""
    ouputlog_filename=""

    def __init__(self, getfaceclass):

        # check current project location path
        currentPath = os.getcwd()
        self.ouputlog_filename = getfaceclass+"_model_arg_log.txt"
        self.output_path= currentPath+'\\log\\model_log\\'+getfaceclass+'\\'      

    def outputmodellog(self,getlog):

        if not os.path.exists(str(self.output_path+self.ouputlog_filename)):
            os.mkdir(str(self.output_path))

        filewrite = open(str(self.output_path+self.ouputlog_filename), 'w')
        filewrite.write(getlog)
        filewrite.close()