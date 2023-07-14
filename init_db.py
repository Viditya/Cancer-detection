import sqlite3

connection = sqlite3.connect('database.db')


with open('schema.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

cur.execute("INSERT INTO tracker (ip, search_text) VALUES (?, ?)",
            ('102.102.03.01', 'Cat.jpg')
            )

cur.execute("INSERT INTO tracker (ip, search_text) VALUES (?, ?)",
            ('102.102.03.02', 'Dog.jpg')
            )

connection.commit()
connection.close()
