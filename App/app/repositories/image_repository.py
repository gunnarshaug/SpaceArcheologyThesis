from db import query_db, store_db


def store_image(path):
    query = 'insert into Image (path) values (?)'
    return store_db(query, [path])


def get_by_id(id):
    query = 'select * from Image where id = ?'
    return query_db(query, [id], one=True)

def delete_image(id):
    query = 'delete from Image where id = ?'
    return store_db(query, [id])


def get_last_id():
    query = "select seq from sqlite_sequence WHERE name = 'Image'"
    response = query_db(query, [], one=True)
    return response['seq']
