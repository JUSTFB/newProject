const http = require('http');
const fs = require('fs');

const results = [];

function test(label, method, path) {
  return new Promise(resolve => {
    const req = http.request({ hostname: 'localhost', port: 5000, path, method, timeout: 5000 }, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        results.push(label + ': ' + res.statusCode + ' -> ' + data.substring(0, 80));
        resolve();
      });
    });
    req.on('error', e => { results.push(label + ': ERROR ' + e.message); resolve(); });
    req.end();
  });
}

(async () => {
  await test('GET /', 'GET', '/');
  await test('POST /api/process', 'POST', '/api/process');
  await test('GET /api/status/x', 'GET', '/api/status/x');
  await test('GET /api/unknown', 'GET', '/api/unknown');
  fs.writeFileSync('D:/newProject/backend/results.txt', results.join('\n'), 'utf8');
})();
