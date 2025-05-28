import React, { useState } from 'react';

const Tema5 = () => {
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleRun = async () => {
    setLoading(true);
    setOutput('');
    try {
      const res = await fetch('http://localhost:5000/tema5-output', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}), // No input needed for default run
      });
      const data = await res.json();
      setOutput(data.output);
    } catch (err) {
      setOutput('Eroare la conectarea cu serverul.');
    }
    setLoading(false);
  };

  return (
    <div>
      <h2>Analiză Matrice Rară simetrică, metoda puterii și SVD (Tema 5)</h2>
      <button onClick={handleRun} disabled={loading} style={{marginBottom: '1em'}}>
        {loading ? 'Se execută...' : 'Rulează Tema 5'}
      </button>
      <pre style={{background:'#222', color:'#fff', padding:'1em', borderRadius:'8px', minHeight:'10em', maxHeight:'60vh', overflowY:'auto'}}>{output}</pre>
    </div>
  );
};

export default Tema5;