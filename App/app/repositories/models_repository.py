from db import query_db, store_db


def get_all():
    query = 'select * from Models order by id desc'
    return query_db(query, [], one=False)


def store_model(title, description, path):
    query = 'insert into Models (title, description, path) values (?, ?, ?)'
    return store_db(query, [title, description, path])


def get_by_id(id):
    query = 'select * from Models where id = ?'
    return query_db(query, [id], one=True)


def delete(id):
    query = 'delete from Models where id = ?'
    return store_db(query, [id])
