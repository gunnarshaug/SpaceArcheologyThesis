from db import store_db, query_db



def delete_mapbox(id):
    query = 'delete from Mapbox where id = ?'
    return store_db(query, [id])



def get_by_id(id):
    query = 'select M.*, I.path from Mapbox M join Image I on I.id = M.image_id where M.id = ?'
    return query_db(query, [id], one=True)


def get_all():
    query = 'select * from Mapbox order by id desc'
    return query_db(query, [], one=False)


def store_mapbox(title, description, coordinates, image_id):
    normalized_coordinates = ';'.join([str(coordinate) for coordinate in coordinates])

    query = 'insert into Mapbox (location, description, coordinates, image_id) values (?, ?, ?, ?)'

    store_db(
        query,
        [title, description, normalized_coordinates, image_id]
    )
