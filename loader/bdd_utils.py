from collections import namedtuple

# a label and all meta information
# Code inspired by Cityscapes https://github.com/mcordts/cityscapesScripts
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',
    # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',
    # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',
    # Whether this label distinguishes between single instances or not

    'ignoreInEval',
    # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

# Labels for the cropped dataset
michael_labels = [
    Label(  'red_light'            ,  0 ,        0 , 'void'            , 0       , False        , False        ,  (0, 0, 0)),
    Label(  'black_person'         ,  1 ,        1 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'red_person'           ,  2 ,        2 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'grey_car'             ,  3 ,        3 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'grey_suv'             ,  4 ,        4 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'green_light'          ,  5 ,        5 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'white_truck'          ,  6 ,        6 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'black_car'            ,  7 ,        7 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'black_bicycle'        ,  8 ,        8 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'grey_person'          ,  9 ,        9 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'white_suv'            , 10 ,       10 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'white_bus'            , 11 ,       11 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'red_suv'              , 12 ,       12 , 'void'            , 0       , False        , False        ,  (0, 0, 0)),  
    Label(  'yellow_car'           , 13 ,       13 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'black_suv'            , 14 ,       14 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'black_truck'          , 15 ,       15 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'white_car'            , 16 ,       16 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'yellow_bus'           , 17 ,       17 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'white_sign'           , 18 ,       18 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'yellow_light'         , 19 ,       19 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'grey_truck'           , 20 ,       20 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'red_car'              , 21 ,       21 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'red_truck'            , 22 ,       22 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'yellow_sign'          , 23 ,       23 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'blue_suv'             , 24 ,       24 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'white_light'          , 25 ,       25 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'yellow_person'        , 26 ,       26 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'white_person'         , 27 ,       27 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'blue_person'          , 28 ,       28 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'red_bus'              , 29 ,       29 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'stop_sign'            , 30 ,       30 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'blue_car'             , 31 ,       31 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'yellow_suv'           , 32 ,       32 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'green_person'         , 33 ,       33 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'black_motorcycle'     , 34 ,       34 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'yellow_truck'         , 35 ,       35 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'green_truck'          , 36 ,       36 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'yellow_bicycle'       , 37 ,       37 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'brown_person'         , 38 ,       38 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'grey_bus'             , 39 ,       39 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'green_sign'           , 40 ,       40 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'blue_bus'             , 41 ,       41 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'black_bus'            , 42 ,       42 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'green_car'            , 43 ,       43 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'green_suv'            , 44 ,       44 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'brown_car'            , 45 ,       45 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'red_sign'             , 46 ,       46 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'white_motorcycle'     , 47 ,       47 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'blue_bicycle'         , 48 ,       48 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
    Label(  'blue_truck'           , 49 ,       49 , 'void'            , 0       , False        , False        ,  (0, 0, 0)), 
]

bdd_noisy_labels = [
    Label(  'unlabeled'            ,  0 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'person'               ,  1 ,        1 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'bicycle'              ,  2 ,        2 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'car'                  ,  3 ,        3 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'motorcycle'           ,  4 ,        4 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'airplane'             ,  5 ,        5 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'bus'                  ,  6 ,        6 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'train'                ,  7 ,        7 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'truck'                ,  8 ,        8 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'boat'                 ,  9 ,        9 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'traffic light'        , 10 ,       10 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'fire hydrant'         , 11 ,       11 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'stop sign'            , 12 ,       12 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'parking meter'        , 13 ,       13 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'bench'                , 14 ,       14 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
]


# Our extended list of label types. Our train id is compatible with Cityscapes
bdd_labels = [
    #       name                     id    trainId   category catId      hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  1 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ego vehicle'          ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ground'               ,  3 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'parking'              ,  5 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           ,  6 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'bridge'               ,  9 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'building'             , 10 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'fence'                , 11 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'garage'               , 12 ,      255 , 'construction'    , 2       , False        , True         , (180,100,180) ),
    Label(  'guard rail'           , 13 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'tunnel'               , 14 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'wall'                 , 15 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'banner'               , 16 ,      255 , 'object'          , 3       , False        , True         , (250,170,100) ),
    Label(  'billboard'            , 17 ,      255 , 'object'          , 3       , False        , True         , (220,220,250) ),
    Label(  'lane divider'         , 18 ,      255 , 'object'          , 3       , False        , True         , (255, 165, 0) ),
    Label(  'parking sign'         , 19 ,      255 , 'object'          , 3       , False        , False        , (220, 20, 60) ),
    Label(  'pole'                 , 20 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 21 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'street light'         , 22 ,      255 , 'object'          , 3       , False        , True         , (220,220,100) ),
    Label(  'traffic cone'         , 23 ,      255 , 'object'          , 3       , False        , True         , (255, 70,  0) ),
    Label(  'traffic device'       , 24 ,      255 , 'object'          , 3       , False        , True         , (220,220,220) ),
    Label(  'traffic light'        , 25 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 26 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'traffic sign frame'   , 27 ,      255 , 'object'          , 3       , False        , True         , (250,170,250) ),
    Label(  'terrain'              , 28 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'vegetation'           , 29 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'sky'                  , 30 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 31 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 32 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'bus'                  , 34 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'car'                  , 35 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'caravan'              , 36 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'motorcycle'           , 37 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'trailer'              , 38 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 39 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'truck'                , 40 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
]
