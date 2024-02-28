
import utils
data = utils.loadDataSet("jamescalam/llama-2-arxiv-papers-chunked")

print(data[0])
# print(data[0]["doi"], data[1]["doi"], data[0]["chunk-id"], data[1]["chunk-id"])