from pic import *

#load relation annotation
gtRel = Relation()
gtRel.loadAnnot('demo_example/pic/relations.json')

gtCatePath ='demo_example/pic/category'
gtInsPath = 'demo_example/pic/instance'

imgname = 'indoor_02018.jpg'

#load segmentation annotation
cateImg = cv2.imread(os.path.join(gtCatePath,imgname.replace('.jpg','.png')), cv2.IMREAD_GRAYSCALE)
insImg = cv2.imread(os.path.join(gtInsPath,imgname.replace('.jpg','.png')), cv2.IMREAD_GRAYSCALE)
img = cv2.imread(os.path.join('demo_example/pic/images',imgname))

#visualize segmentation
visualize_segmentation(cateImg, img)

#visualize instance
visualize_instance(insImg, cateImg, img)

relation = gtRel.loadRel(imgname)
#visualize relation
visualize_relation(relation,insImg,img)

#evalute code is coming soon
