import json

def com(json_file, json_file2, json_file3, out_file):
    file = open(json_file, 'r')
    json_list = json.load(file)
    
    file2 = open(json_file2, 'r')
    json_list2 = json.load(file2)
    
    file3 = open(json_file3, 'r')
    json_list3 = json.load(file3)
    
    res = json_list + json_list2 + json_list3

    with open(out_file, 'w') as f:
        json.dump(res, f)
        
if __name__ == '__main__':
    com('books_result.json', 'books_result_page_2.json', 'books_result_page_3.json', 'books_result_page_2000.json')