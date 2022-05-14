from db import query_db, store_db


def get_all():
    query = 'select * from Results join Models M on M.id = Results.model_id order by id desc'
    return query_db(query, [], one=False)


def store_results(model_id, path):
    query = 'insert into Results (model_id, path) values (?, ?)'
    return store_db(query, [model_id, path])


def get_by_id(id):
    query = 'select * from Results join Models M on M.id = Results.model_id where id = ?'
    return query_db(query, [id], one=True)


def delete(id):
    query = 'delete from Results where id = ?'
    return store_db(query, [id])
