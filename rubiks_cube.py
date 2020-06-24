from rubik.cube import Cube
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.utils
import itertools

SOLVED_STATE = "RRRRRRRRRBBBWWWGGGYYYBBBWWWGGGYYYBBBWWWGGGYYYOOOOOOOOO"
actions = ('L', 'Li', 'R', 'Ri', 'F', 'Fi', 'B', 'Bi', 'U', 'Ui', 'D', 'Di')

corners = ('GOW', 'GOB', 'GRW', 'GRB', 'YOW', 'YOB', 'YRW', 'YRB')

edges = ('GO', 'GR', 'GW', 'GB', 'YO', 'YR', 'YW', 'YB', 'OW', 'OB', 'RW', 'RB')

'''
starting from solved state, randomly move 100 times to get 100 different states and the depth associated. the depth will be turned 
into a weight to the loss later during the training.

for each state in the set, do breadth-1 search and get the 12 different sub states.

inference nn to get the policy and value of the parent node
inference all 12 sub nodes to get each children value and children's policy

loss =  
batch_loss = sum(weight * loss)
'''
def data_generator():
    c = Cube(SOLVED_STATE)
    print_cube(c)
    for x in np.random.randint(0,12,100):
        getattr(c, actions[x])() #scramble cube
        print(f"action={actions[x]}")
        print_cube(c)
        str = c.flat_str()
        print("12 children")
        for a in actions:
            c_working = Cube(str)
            getattr(c_working, a)()
            print_cube(c_working)
        print("--------------------")


class bcolors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    WHITE = '\033[37m'
    MAGENTA = '\033[35m'
    BLACK = '\033[30m'

def terminal_str(s):
    s = s.replace("R", bcolors.RED+"R")
    s = s.replace("G", bcolors.GREEN+"G")
    s = s.replace("B", bcolors.BLUE+"B")
    s = s.replace("W", bcolors.WHITE+"W")
    s = s.replace("O", bcolors.MAGENTA+"O")
    s = s.replace("Y", bcolors.YELLOW+"Y")
    return s

def print_cube(c):
    s = c.flat_str()

    index = 0
    for i in range(3):
        print(" "*3 + " " + terminal_str(s[index:index+3]))
        index += 3

    for i in range(3):
        for j in range(4):
            print(terminal_str(s[index:index+3] + " "), end='')
            index += 3
        print('')

    for i in range(3):
        print(" "*3 + " " + terminal_str(s[index:index+3]))
        index += 3
    print('')

def create_model():
    gpu_options = tf.compat.v1.GPUOptions()
    gpu_options.allow_growth = True


def build_model():
    inputs = keras.Input(shape=(20,24), dtype=tf.int32)

    x = keras.layers.Reshape(target_shape=(20*24,), input_shape=(20,24))(inputs)
    x = keras.layers.Dense(units=4096, activation='elu')(x)
    common = keras.layers.Dense(units=2048, activation='elu')(x)

    policy_x = keras.layers.Dense(units=512, activation='elu')(common)
    policy_x = keras.layers.Dense(units=12, activation='softmax')(policy_x)

    value_x = keras.layers.Dense(units=512, activation='elu')(common)
    value_x = keras.layers.Dense(units=1, activation='relu')(value_x)

    return keras.Model(inputs=inputs, outputs=[policy_x, value_x])

def train(x, y_action, y_value):
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    action_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    value_loss_fn = keras.losses.mean_squared_error()
    #with tf.GradientTape() as tape:
    #    grads = tape.gradient(loss_value, model.trainable_variables)
    #    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def reformat_data(cube_str):
    pass


def test_dict():
    c = Cube(SOLVED_STATE)
    edges = [''.join(filter(None, x.colors)) for x in c.pieces if x.type == 'edge']
    edges = [[x, x[1:]+x[:1]] for x in edges]
    edges = [x for sublist in edges for x in sublist]
    edges = {edges[i]:i for i in range(len(edges))}

    corners = [''.join(x.colors) for x in c.pieces if x.type == 'corner']
    corners = [(x, x[0]+x[2]+x[1], x[2]+x[1]+x[0]) for x in corners]
    corners = [x for sublist in corners for x in sublist]
    corners = {corners[i]: i for i in range(len(corners))}

    for x in np.random.randint(0,12,100):
        getattr(c, actions[x])() #scramble cube
        print([edges[m] for m in [''.join(filter(None, x.colors)) for x in c.pieces if x.type == 'edge']])
        print([corners[m] for m in [''.join(filter(None, x.colors)) for x in c.pieces if x.type == 'corner']])

def init_lookup(c):
    es = []
    edges = [x[1] if x[0] > 0 else x[1].lower() for p in c.pieces if p.type == 'edge' for x in zip(p.pos, [v if v!=None else '_' for v in p.colors])]
    edges = [''.join(x) for x in zip(edges[0::3], edges[1::3], edges[2::3])]
    for e in edges:
        es.append([c for v in [''.join(x) for x in itertools.permutations(e)] for c in case_permutations(v)])
    es = np.array(es)

    cnr = []
    for i in range(2):
        for i in range(4): #rotate the top and bottom 4 times to get all different corner colors
            corners = [x[1] if x[0] > 0 else x[1].lower() for p in c.pieces if p.type == 'corner' for x in
                       zip(p.pos, p.colors)]
            cnr.append([''.join(x) for x in zip(corners[0::3], corners[1::3], corners[2::3])])
            corners = [x[1] if x[0] > 0 else x[1].lower() for p in c.pieces if p.type == 'corner' for x in
                       zip(p.pos, p.colors[1:]+p.colors[:1])]
            cnr.append([''.join(x) for x in zip(corners[0::3], corners[1::3], corners[2::3])])
            corners = [x[1] if x[0] > 0 else x[1].lower() for p in c.pieces if p.type == 'corner' for x in
                       zip(p.pos, p.colors[2:]+p.colors[:2])]
            cnr.append([''.join(x) for x in zip(corners[0::3], corners[1::3], corners[2::3])])
            c.U()
            c.D()
        c.Z()
        c.Z()
    cnr = np.array(cnr)
    cnr = cnr.transpose()

    lookup = {}
    for z in es:
        for x in range(len(z)):
            lookup[z[x]] = x

    for z in cnr:
        for x in range(len(z)):
            lookup[z[x]] = x

    return lookup

def case_permutations(str):
    import itertools
    if not str:
        yield ""
    else:
        x = str[:1]
        if x.lower() == x.upper():
            for y in case_permutations(str[1:]):
                yield x+y
        else:
            for y in case_permutations(str[1:]):
                yield x.lower() + y
                yield x.upper() + y

if __name__ == "__main__":
    model = build_model()
    print(model.summary())
    #state = np.random.randint(0, 24, (10,20))
    #state = tf.keras.utils.to_categorical(state, 24)
    #actions, value = model.predict(state)
    #print(model)

    c = Cube(SOLVED_STATE)

    lookup = init_lookup(c)
    corners = [''.join(p.colors) for p in c.pieces if p.type == 'corner']
    corner_dict = {k: v for v, k in enumerate(corners)}

    for x in np.random.randint(0,12,200):
        getattr(c, actions[x])() #scramble cube
        corners = [x[1] if x[0] > 0 else x[1].lower() for p in c.pieces if p.type == 'corner' for x in
                   zip(p.pos, p.colors)]
        corners = [''.join(x) for x in zip(corners[0::3], corners[1::3], corners[2::3])]

        corners.sort(key=lambda x: int(x[0].lower() == x[0]) * 4 + int(x[1].lower() == x[1]) * 2 + int(x[2].lower() == x[2]))

        state = [lookup[y] for y in corners]

        edges = [x[1] if x[0] > 0 else x[1].lower() for p in c.pieces if p.type == 'edge' for x in zip(p.pos, [v if v!=None else '_' for v in p.colors])]
        edges = [''.join(x) for x in zip(edges[0::3], edges[1::3], edges[2::3])]
        state.extend([lookup[y] for y in edges])

        state = [state]
        state = tf.keras.utils.to_categorical(state, 24)
        actions, value = model.predict(state)

    pass

#Corners
#('Wgr', 'GwR', 'Grw', 'RgW', 'WrG', 'GWr', 'rWg', 'gRw', 'RGw', 'rGW', 'GRW', 'wrg', 'WGR', 'rgw', 'wGr', 'gWR', 'rwG', 'wgR', 'Rwg', 'gwr', 'RWG', 'grW', 'WRg', 'wRG')
#{'gOy', 'OgY', 'Oyg', 'GyO', 'GOY', 'goY', 'yGo', 'GYo', 'Goy', 'OGy', 'ygO', 'YOg', 'ogy', 'yog', 'oGY', 'yOG', 'YGO', 'OYG', 'oyG', 'gYO', 'YoG', 'Ygo', 'oYg', 'gyo'}
#{'BYr', 'Bry', 'RYB', 'Ybr', 'byr', 'yBr', 'YBR', 'yRB', 'rby', 'Ryb', 'RBy', 'YRb', 'yrb', 'BRY', 'ByR', 'bRy', 'rYb', 'ybR', 'RbY', 'rBY', 'brY', 'YrB', 'ryB', 'bYR'}
#{'bOw', 'bWO', 'Owb', 'boW', 'BwO', 'WoB', 'BWo', 'WBO', 'bwo', 'OWB', 'oBW', 'BOW', 'Wbo', 'obw', 'ObW', 'wBo', 'wOB', 'OBw', 'owB', 'Bow', 'wob', 'oWb', 'WOb', 'wbO'}
#{'RBW', 'RWb', 'WBr', 'bRW', 'wBR', 'wRb', 'wrB', 'WRB', 'rwb', 'RwB', 'WbR', 'Wrb', 'bwR', 'brw', 'Rbw', 'rBw', 'rWB', 'BWR', 'bWr', 'rbW', 'BrW', 'BRw', 'wbr', 'Bwr'}
#{'oyb', 'ybo', 'yOb', 'OBY', 'bOY', 'obY', 'byO', 'BYO', 'bYo', 'boy', 'YbO', 'oBy', 'Byo', 'YOB', 'yBO', 'Yob', 'Oby', 'oYB', 'BOy', 'OYb', 'BoY', 'OyB', 'yoB', 'YBo'}
#{'GRy', 'YRG', 'rYG', 'RYg', 'rGy', 'Yrg', 'Gyr', 'YgR', 'ygr', 'ryg', 'RyG', 'gYr', 'gRY', 'RGY', 'Rgy', 'yrG', 'GYR', 'gry', 'GrY', 'yRg', 'gyR', 'yGR', 'rgY', 'YGr'}
#{'oGw', 'gow', 'WOG', 'woG', 'Ogw', 'oWG', 'Wog', 'WgO', 'gOW', 'OWg', 'wOg', 'ogW', 'wgo', 'gwO', 'GoW', 'OwG', 'Gwo', 'GWO', 'owg', 'WGo', 'OGW', 'gWo', 'GOw', 'wGO'}

#edges
#for x in e:
#    print([v+"_" for v in all_casings(x)] + [v[0]+"_"+v[1] for v in all_casings(x)] + ["_"+v for v in all_casings(x)]
#         +[v+"_" for v in all_casings(x[::-1])] + [v[0]+"_"+v[1] for v in all_casings(x[::-1])] + ["_"+v for v in all_casings(x[::-1])])

#['gr_', 'Gr_', 'gR_', 'GR_', 'g_r', 'G_r', 'g_R', 'G_R', '_gr', '_Gr', '_gR', '_GR', 'rg_', 'Rg_', 'rG_', 'RG_', 'r_g', 'R_g', 'r_G', 'R_G', '_rg', '_Rg', '_rG', '_RG']
#['go_', 'Go_', 'gO_', 'GO_', 'g_o', 'G_o', 'g_O', 'G_O', '_go', '_Go', '_gO', '_GO', 'og_', 'Og_', 'oG_', 'OG_', 'o_g', 'O_g', 'o_G', 'O_G', '_og', '_Og', '_oG', '_OG']
#['gw_', 'Gw_', 'gW_', 'GW_', 'g_w', 'G_w', 'g_W', 'G_W', '_gw', '_Gw', '_gW', '_GW', 'wg_', 'Wg_', 'wG_', 'WG_', 'w_g', 'W_g', 'w_G', 'W_G', '_wg', '_Wg', '_wG', '_WG']
#['gy_', 'Gy_', 'gY_', 'GY_', 'g_y', 'G_y', 'g_Y', 'G_Y', '_gy', '_Gy', '_gY', '_GY', 'yg_', 'Yg_', 'yG_', 'YG_', 'y_g', 'Y_g', 'y_G', 'Y_G', '_yg', '_Yg', '_yG', '_YG']
#['br_', 'Br_', 'bR_', 'BR_', 'b_r', 'B_r', 'b_R', 'B_R', '_br', '_Br', '_bR', '_BR', 'rb_', 'Rb_', 'rB_', 'RB_', 'r_b', 'R_b', 'r_B', 'R_B', '_rb', '_Rb', '_rB', '_RB']
#['bo_', 'Bo_', 'bO_', 'BO_', 'b_o', 'B_o', 'b_O', 'B_O', '_bo', '_Bo', '_bO', '_BO', 'ob_', 'Ob_', 'oB_', 'OB_', 'o_b', 'O_b', 'o_B', 'O_B', '_ob', '_Ob', '_oB', '_OB']
#['bw_', 'Bw_', 'bW_', 'BW_', 'b_w', 'B_w', 'b_W', 'B_W', '_bw', '_Bw', '_bW', '_BW', 'wb_', 'Wb_', 'wB_', 'WB_', 'w_b', 'W_b', 'w_B', 'W_B', '_wb', '_Wb', '_wB', '_WB']
#['by_', 'By_', 'bY_', 'BY_', 'b_y', 'B_y', 'b_Y', 'B_Y', '_by', '_By', '_bY', '_BY', 'yb_', 'Yb_', 'yB_', 'YB_', 'y_b', 'Y_b', 'y_B', 'Y_B', '_yb', '_Yb', '_yB', '_YB']
#['rw_', 'Rw_', 'rW_', 'RW_', 'r_w', 'R_w', 'r_W', 'R_W', '_rw', '_Rw', '_rW', '_RW', 'wr_', 'Wr_', 'wR_', 'WR_', 'w_r', 'W_r', 'w_R', 'W_R', '_wr', '_Wr', '_wR', '_WR']
#['ry_', 'Ry_', 'rY_', 'RY_', 'r_y', 'R_y', 'r_Y', 'R_Y', '_ry', '_Ry', '_rY', '_RY', 'yr_', 'Yr_', 'yR_', 'YR_', 'y_r', 'Y_r', 'y_R', 'Y_R', '_yr', '_Yr', '_yR', '_YR']
#['ow_', 'Ow_', 'oW_', 'OW_', 'o_w', 'O_w', 'o_W', 'O_W', '_ow', '_Ow', '_oW', '_OW', 'wo_', 'Wo_', 'wO_', 'WO_', 'w_o', 'W_o', 'w_O', 'W_O', '_wo', '_Wo', '_wO', '_WO']
#['oy_', 'Oy_', 'oY_', 'OY_', 'o_y', 'O_y', 'o_Y', 'O_Y', '_oy', '_Oy', '_oY', '_OY', 'yo_', 'Yo_', 'yO_', 'YO_', 'y_o', 'Y_o', 'y_O', 'Y_O', '_yo', '_Yo', '_yO', '_YO']

