import { useEffect, useState, type FormEvent } from "react";
import { signIn, signInOptions, type Session, type SignInOption } from "./api";

// a fake sign-in: the simulated business hands out a session for any of its made-up customers, no password
export function SignIn({ onSignedIn }: { onSignedIn: (session: Session) => void }) {
  const [options, setOptions] = useState<SignInOption[] | null>(null);
  const [typed, setTyped] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [working, setWorking] = useState(false);

  useEffect(() => {
    signInOptions()
      .then(setOptions)
      .catch(() => {
        setOptions([]);
        setError("Couldn't reach the HiveMart backend. Is it running?");
      });
  }, []);

  const enter = async (customerId: string) => {
    setWorking(true);
    setError(null);
    try {
      onSignedIn(await signIn(customerId));
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : "Sign-in failed.");
      setWorking(false);
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const id = typed.trim().toUpperCase();
    if (id) void enter(id);
  };

  return (
    <main className="sign-in">
      <h1>HiveMart support</h1>
      <p className="note">
        Demo sign-in. Every customer here is made up and no password is asked. Pick one to open their support chat.
      </p>

      {options === null && <p className="muted">Loading customers…</p>}
      <ul className="customers">
        {(options ?? []).map((option) => (
          <li key={option.customer_id}>
            <button type="button" disabled={working} onClick={() => void enter(option.customer_id)}>
              <span className="name">{option.name}</span>
              <span className="id">{option.customer_id}</span>
              <span className="hints">{option.hints.join(" · ")}</span>
            </button>
          </li>
        ))}
      </ul>

      <form onSubmit={submit} className="by-id">
        <label htmlFor="customer-id">Or type a customer id</label>
        <div className="row">
          <input
            id="customer-id"
            value={typed}
            onChange={(event) => setTyped(event.target.value)}
            placeholder="CUST-00042"
            autoComplete="off"
          />
          <button type="submit" disabled={working || !typed.trim()}>
            Sign in
          </button>
        </div>
      </form>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </main>
  );
}
