from bottle import get, post, request,run # or route
from bottle import template,route,Bottle
from bottle import static_file
'''
@route('/hello')
def hello():
    return "Hello World!"
run(host='localhost', port=8080, debug=True)
'''

app=Bottle()
'''
@app.route('/hello')
def hello():
    return "Hello World!"
    '''


'''
@route('/')
@route('/hello/<name>')
def greet(name='Stranger'):
    return template('Hello {{name}}, how are you?', name=name)
    '''
'''
def check_login(u,p):
    if u== p:
        return True
    else:
        return False
@get('/login')
def login():
    return ''''''
        <form action="/login" method="post">
            username:<input name="username" type="text"/>
            password:<input name="password" type="password"/>
            <input value="Login" type="submit"/>
        </form>
''''''
@post('/login')
def do_login():
    username = request.forms.get('username')
    password = request.forms.get('password')
    if check_login(username,password):
        return "<p>Your login information was correct.<p>"
    else:
        return "<p>login failed</p>"

        '''
#run(host='localhost', port=8080)

'''
from bottle import static_file

@route('/static/<filename>')
def server_static(filename):
    return static_file(filename,root='')


run(host='localhost', port=8080)
'''


from bottle import request,route,template
@route('/index')
def index():
    return template("index",name = 'nikhil')

@route('/images/<folder>/<filename:re:.*\.jpg>')
def send_image(filename,folder):
    return static_file(filename, root='E:/Nikhil/python/videoclass/videoclass/test/{}'.format(folder), mimetype='image/jpg')

@route('/static/<filename:path>')
def send_static(filename):
    return static_file(filename, root='E:/Nikhil/python/videoclass/videoclass/test/')

@get('/tell1')

def get2():
    return '''<form action="/tell1" method="post" >
    Select image to upload:
    <input type="text" webkitdirectory mozdirectory name="fileToUpload" id="fileToUpload">
    
    Enter the class folder of the images to predict:
    <input type="text" webkitdirectory mozdirectory name="classes" id="fileToUpload">
    
    Enter the number of images present in the folder(Batch_Size):
    <input type="number" webkitdirectory mozdirectory name="Batch_size" id="fileToUpload">
    <input type="submit" value="Upload Image" name="submit">
    
</form>'''


@post('/tell1')
def get1():
    filepath=request.forms.get('fileToUpload')
    filepath1=request.forms.get('classes')
    filepath2=request.forms.get('Batch_size')
    print(filepath)
    filepath2=int(filepath2)
    if filepath == filepath:
        
        return template("index",name=filepath,name1=filepath1,name2=filepath2)


    
run(host='localhost', port=8080)




























