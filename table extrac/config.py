import numpy as np
from PIL import Image, ImageDraw
import cv2

class econfig:
    def __init__(self):
        
        self.col_edge=[ 353,  568,  763,  832, 1930, 2089]
        self.row_edge=[ 355,  486,  682,  873,  911,  948,  988, 1026]
        
        self.crop_row = [486, 531, 569, 606, 645, 683, 721, 759, 797, 835, 873]
        self.crop_col = [ 763,  832,  889,  947, 1005, 1063, 1120, 1178, 1236, 1294, 1352, 1409, 1467, 1525, 1583, 1641, 1698, 1756, 1814, 1872, 1930]
        
        
       
        
        # self.image = image.convert('RGB')
        # self.point = point
        
        #draw line
        # img1 = ImageDraw.Draw(self.image)   
        # for i in range(len(self.col_edge)):
        #     img1.line([(self.col_edge[i],0),(self.col_edge[i],self.image.size[1])], fill ="red", width = 3) 
        # for i in range(len(self.row_edge)):
        #     img1.line([(0,self.row_edge[i]),(self.image.size[0],self.row_edge[i])], fill ="red", width = 3) 
            
        # #draw verticle
        # self.crop_col=[self.col_edge[2]]
        # x=19
        # for i in range(x+1):
        #     self.crop_col.append(self.col_edge[3]*(1-i/x)  + self.col_edge[4]*(i/x))
        #     # img1.line([(self.col_edge[3]*(i/x) + self.col_edge[4]*(1-i/x) ,0),(self.col_edge[3]*(i/x) + self.col_edge[4]*(1-i/x)  ,self.image.size[1])], fill ="green", width = 3) 
        #     # img1.line([(self.col_edge[2]*(i/x) + self.col_edge[3]*(1-i/x) ,0),(self.col_edge[2]*(i/x) + self.col_edge[3]*(1-i/x)  ,self.image.size[1])], fill ="green", width = 3) 

        # # #draw horizon
        # self.crop_row=[]
        # x=10
        # for i in range(x+1):
        #     self.crop_row.append(self.row_edge[1]*(1-i/x) + self.row_edge[3]*(i/x))
        #     # img1.line([(0,self.row_edge[1]*(i/x) + self.row_edge[3]*(1-i/x)),(self.image.size[0],self.row_edge[1]*(i/x) + self.row_edge[3]*(1-i/x))], fill ="green", width = 3) 

        # print((self.col_edge[0],self.row_edge[0],self.col_edge[1],self.row_edge[1]))
        # print(self.col_edge)
        # print(self.row_edge)
        
        # draw crop_data
        # self.crop_row=np.asarray(self.crop_row).astype('int')
        # self.crop_col=np.asarray(self.crop_col).astype('int')
        # print(self.crop_row)
        # print(self.crop_col)
        # for i in range(len(self.crop_col)):
        #     img1.line([(self.crop_col[i],self.crop_row[0]),(self.crop_col[i],self.crop_row[-1])], fill ="blue", width = 3) 
        # for i in range(len(self.crop_row)):
        #     img1.line([(self.crop_col[0],self.crop_row[i]),(self.crop_col[-1],self.crop_row[i])], fill ="blue", width = 3) 
        
        # self.data=[]
        # self.data.append(
        #     {
        #         "idnum":image.crop((int(self.col_edge[1]),int(self.row_edge[1]),int(self.col_edge[2]),int(self.row_edge[2]))),
        #         "name":image,
        #         "score":image
        #     } )
        
    def draw_line(self,image):
        image = image.convert('RGB')
        img1 = ImageDraw.Draw(image)  
        for i in range(len(self.crop_col)):
            img1.line([(self.crop_col[i],self.crop_row[0]),(self.crop_col[i],self.crop_row[-1])], fill ="blue", width = 3) 
        for i in range(len(self.crop_row)):
            img1.line([(self.crop_col[0],self.crop_row[i]),(self.crop_col[-1],self.crop_row[i])], fill ="blue", width = 3) 
        
        return image
    
    def crop_data(self,image,extend=0):
        image = image.convert('RGB')
        index=0
        result=[]
        for row_index in range(len(self.crop_row)-1):
             for col_index in range(len(self.crop_col)-1):
                 result.append({
                     "image":image.crop(( self.crop_col[col_index]-extend, self.crop_row[row_index]-extend,self.crop_col[col_index+1]+extend,  self.crop_row[row_index+1]+extend)),
                     "index":index,
                     "location":(self.crop_col[col_index],self.crop_row[row_index],self.crop_col[col_index+1], self.crop_row[row_index+1])
                     
                     })
        return result
        
if __name__ == "__main__":
    # from PIL import Image
    image = Image.open(r"/media/trainai/data/project/python/e-election/FeatureMatching/output/5.jpg")

    # ec = econfig(image,[[832,915],[3136,902],[3148,1840],[824,1831]]) #w/h => 4032/3024
    ec = econfig() #w/h => 4032/3024
    # ec.image.show()
    ec.draw_line(image).save('djs.png')
    #ref point 304,303   2149,303   2149,1039 304,1039 w/h =>2339,1653
    data = ec.crop_data(image,0)
    for i in range(len(data)):
        data[i]['image'].save('/media/trainai/data/db/election/Score/train/0010000/'+str(i)+'a.png')
