import React, { useEffect, useState } from 'react';

const Tema1 = () => {
  const [exercises, setExercises] = useState([]);

  useEffect(() => {
    fetch('http://localhost:5000/tema1-output')
      .then(res => res.json())
      .then(data => {
        console.log(data.output);
        const parsed = parseExercises(data.output);
        setExercises(parsed);
      })
      .catch(err => {
        setExercises([{ title: 'Eroare', content: err.message }]);
      });
  }, []);

  const parseExercises = (output) => {
    const lines = output.split('\n');
    const result = [];

    let currentTitle = null;
    let currentContent = [];

    const isExerciseTitle = (line) =>
      /^ex\d+[a-z]?$/.test(line.trim()); // matches ex1, ex2a, ex3, etc.

    for (const line of lines) {
      if (isExerciseTitle(line)) {
        if (currentTitle) {
          result.push({ title: currentTitle, content: currentContent.join('\n') });
        }
        currentTitle = line.trim();
        currentContent = [];
      } else {
        currentContent.push(line);
      }
    }

    // Push the last exercise
    if (currentTitle) {
      result.push({ title: currentTitle, content: currentContent.join('\n') });
    }

    return result;
  };

  return (
    <div>
      <h2>Rezultate Tema 1 - Calcul Numeric</h2>
      {exercises.map((ex, idx) => (
        <div key={idx} style={{ marginBottom: '1.5em' }}>
          <h3 style={{ color: 'aquamarine', marginBottom: '0.5em' }}>{ex.title}</h3>
          <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize:'18px',backgroundColor: '#222', padding: '1em', borderRadius: '8px' }}>
            {ex.content}
          </pre>
        </div>
      ))}
    </div>
  );
};

export default Tema1;
