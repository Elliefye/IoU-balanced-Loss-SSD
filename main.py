import model.dataset as d
from model.utils import generate_dboxes, Encoder
from model.transform import SimpleTransformer


dboxes = generate_dboxes()
dataset = d.OIDataset(SimpleTransformer(dboxes))
print(dataset[0])
