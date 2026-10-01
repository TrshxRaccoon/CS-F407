connected(a,b).
connected(b,c).
connected(c,d).

valid_move(X,Y) :- connected(X,Y).

valid_move(X,Y) :-
    connected(X,Z),
    valid_move(Z,Y).