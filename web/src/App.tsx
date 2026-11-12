import { useState } from "react";
import type { Session } from "./api";
import { Chat } from "./Chat";
import { SignIn } from "./SignIn";

const SESSION_KEY = "hivemind.session";

// the session lives in this browser tab only, so a reload keeps the chat and closing the tab signs out
function savedSession(): Session | null {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    const session = raw ? (JSON.parse(raw) as Session) : null;
    return session && new Date(session.expires_at) > new Date() ? session : null;
  } catch {
    return null;
  }
}

export function App() {
  const [session, setSession] = useState<Session | null>(savedSession);

  const signedIn = (next: Session) => {
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(next));
    setSession(next);
  };
  const signOut = () => {
    sessionStorage.clear();
    setSession(null);
  };

  return session ? <Chat session={session} onSignOut={signOut} /> : <SignIn onSignedIn={signedIn} />;
}
