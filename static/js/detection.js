
  var loadFile = function(event) {
    var output = document.getElementById('output');
    output.style.visibility = 'visible';

    var temp  = URL.createObjectURL(event.target.files[0]);
    output.src = temp;

    console.log(output);
    console.log(temp)
    output.onload = function() {
        URL.revokeObjectURL(output.src) // Need to do after upload

        document.getElementById('txt').style.visibility = 'visible';
        document.getElementById('Submit').style.visibility = 'visible';

        change('Title_sentence','Successful upload')

        }
  };


change = function(id,word){
    var elem = document.getElementById(id);
    elem.innerHTML = word

}




function isFileImage(file) {
return file && file['type'].split('/')[0] === 'image';
}

