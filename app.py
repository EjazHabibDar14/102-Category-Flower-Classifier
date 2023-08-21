from fastai.vision.all import *
import gradio as gr
import os

def is_flower_category(x):
    return x[0].issuper()

lis = [i for i in range(101)]
os.mkdir("newclass")
for i in lis:
    os.mkdir('newclass/{}'.format(i))


# Define a function to get the image files
def get_image_files(path, folders=None):
    return get_files(path, extensions='.jpg', recurse=True)

# Create a function to get the label from the image file path
def get_label(file_path):
    return int(file_path.parent.name)

# Define the path to the folder containing your images (class folder)
image_folder_path = '/content/newclass'

learn = load_learner('model.pkl')

categories = (
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris",
    "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss",
    "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
    "blanket flower", "trumpet creeper", "blackberry lily"
)

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    category = categories[idx]
    return dict(zip(categories,map(float,probs)))

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()

examples = ['image_07751.jpg', 'image_04897.jpg', 
            'image_02553.jpg', 'image_06930.jpg', 
            'image_01964.jpg', 'image_04840.jpg']

title = "102 Category Flower Classifier"

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples, title=title)
intf.launch(inline=False)