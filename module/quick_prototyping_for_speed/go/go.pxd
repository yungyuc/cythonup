#cython: infer_types = True, cdivision = True, boundscheck=False
cimport cython
from libc.math cimport sqrt, log
from libc.stdint cimport uint64_t
cdef enum:
    _SIZE = 9
cdef:
    int SIZE, GAMES, EMPTY, WHITE, BLACK, PASS, MAXMOVES
    int TIMESTAMP, MOVES
    double KOMI
    Board board
    int pos

cdef inline int to_pos(int x,int y)
cdef inline tuple to_xy(int pos)
cdef class Square:
    cdef:
        Board board
        int pos, color, used, ledges, temp_ledges
        int timestamp, removestamp
        uint64_t[3] zobrist_strings
        list neighbours
        Square reference
    @cython.locals(x='int', y='int', dx='int', dy='int', newx='int', newy='int')
    cdef void set_neighbours(Square self)
    @cython.locals(neighbour=Square, neighcolor='int', neighbour_ref=Square)
    cdef void move(Square self, int color)
    @cython.locals(neighbour=Square, neighcolor='int', neighbour_ref=Square)
    cdef void remove(Square self, Square reference, int update=*)
    cdef Square find(Square self, int update=*)

cdef class EmptySet:
    cdef:
        Board board
        list empties
        list empty_pos
    @cython.locals(i='int', choices='int')
    cdef int random_choice(self)
    cdef void add(self, int pos)
    cdef void remove(self, int pos)
    cdef void set(self, int i, int pos)

cdef class ZobristHash:
    cdef:
        Board board
        set hash_set
        uint64_t hash
    @cython.locals(square=Square)
    cdef void init(self)
    cdef void update(self, Square square, int color)
    cdef inline void add(self)        
    cdef inline int dupe(self)
    
cdef class Board:
    cdef:
        list squares
        EmptySet emptyset
        ZobristHash zobrist
        int color
        int finished
        int lastmove
        list history
        int white_dead, black_dead
    @cython.locals(square=Square)
    cdef init(self)
    @cython.locals(square=Square)
    cdef reset(self)
    @cython.locals(square=Square)
    cdef move(self, int pos)
    cdef int random_move(self)
    @cython.locals(neighbour=Square)
    cdef useful_fast(self, Square square)
    @cython.locals(square=Square, neighbour=Square, empties ='int',
            opps ='int', weak_opps = 'int',  neighs = 'int', weak_neighs='int')
    cdef useful(self, int pos)
    cdef list useful_moves(self)
    @cython.locals(pos='int')
    cdef void replay(self, list history)
    @cython.locals(square=Square, neighbour=Square, surround='int')
    cdef double score(self, int color)
    @cython.locals(square=Square, neighbour=Square, member=Square, 
    empties1=set, empties2=set, members2=set, ledges1='int')
    cdef void check(self)

cdef class UCTNode:
    cdef:
        UCTNode bestchild
        int pos
        int wins
        int losses
        list pos_child
        UCTNode parent
        list unexplored
    @cython.locals(child=UCTNode, path=list)
    cdef void play(self, Board board)
    @cython.locals(i='int', pos='int')
    cdef int select(self, Board board)
    cdef void random_playout(self, Board board)
    @cython.locals(node=UCTNode, wins='int')
    cdef void update_path(self, Board board, int color, list path)
    @cython.locals(winrate='double', nodevisits='int')
    cdef double score(self)
    @cython.locals(child=UCTNode)
    cdef UCTNode best_child(self)
    @cython.locals(child=UCTNode)
    cdef UCTNode best_visited(self)
    
cdef int user_move(Board board)
@cython.locals(game='int', pos='int', tree=UCTNode)
cdef int computer_move(Board board)
