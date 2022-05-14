import click
from flask import current_app, g
import sqlite3

from flask.cli import with_appcontext


def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

def store_db(query, args=()):
    db = get_db()
    cur = db.execute(query, args)
    rv = cur.fetchall()
    cur.close()
    db.commit()


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            'development.db',
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        g.db.row_factory = sqlite3.Row

    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()


def init_db():
    connection = sqlite3.connect('development.db')
    with open('db/schema.sql') as f:
        connection.executescript(f.read())

    with open('db/static.sql') as f:
        connection.executescript(f.read())



@click.command('init-db')
@with_appcontext
def init_db_command():
    init_db()
    click.echo('Initialized the database.')


def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)

