from datetime import datetime


class Results:

    _hr = '--------------------------------------------------------------------------------------\n'
    _patchOut = ''
    _title = ''
    _description = ''
    _date = ''
    _body = ''
    _results = ''

    def __init__(self, patch_out):
        self._patchOut = patch_out

    def setTitle(self, title):
        self._title = title

    def setDescription(self, description):

        self._description = description

    def setPathOut(self, patch_out):
        self._patchOut = patch_out

    def insertProcessSpace(self, title):
        self._body += self._hr + title + '\n' + self._hr + '\n'

    def insertProcess(self, title, output):
        self._body += '* ' + title + ':  ' + output + '\n'

    def insertResultSpace(self, title):
        self._results += '\n' + self._hr + self._hr + title + '\n' + self._hr + self._hr

    def insertResult(self, title, scoreCV):
        self._results += '(' + title + ') ->' + '\t' + str(scoreCV.mean()) + '\t(+/-) ' + str(scoreCV.std() * 2) + '\n'

    def insertResultTimeMean(self, title, time_mean):
        self._results += '(' + title + ') ->' + '\t' + str(time_mean) +'\n'

    def insertResultAcuracy(self, method, scoreCV):
        self._body += '(' + method + ') ->'+'\t' + str(scoreCV.mean()) + '\t(+/-) ' + str(scoreCV.std()*2) + '\n'

    def save(self):
        date = datetime.now()
        date_formated = str(date.day) + '-' + str(date.month) + '-' + str(date.year) + '  ' + str(date.hour) + ' ' + str(
            date.minute) + ' ' + str(date.second)

        with open(self._patchOut + self._title + date_formated + '.txt', 'w+') as arq:

            arq.write(self._hr + '\n')

            arq.write('++Title: ' + self._title + '\n')
            arq.write('++Description: ' + self._description + '\n')
            arq.write('++Autor: Kreisler Brenner Mendes' + '\n')
            arq.write('++Date: ' + date_formated + '\n')

            arq.write(self._hr+'\n')

            arq.write(self._body)
            arq.write(self._results)



